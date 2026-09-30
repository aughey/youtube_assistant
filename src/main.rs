use std::{
    collections::HashSet,
    fs::File,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use async_openai::{
    Client as OpenAiClient,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
};
use clap::Parser;
use google_youtube3::{
    YouTube, hyper_rustls, hyper_util,
    yup_oauth2::{self, InstalledFlowAuthenticator, InstalledFlowReturnMethod},
};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, time::sleep};
use tracing::{debug, error, info, trace};
use tracing_error::ErrorLayer;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

const DEFAULT_CATEGORY_ID: &str = "22";
const DEFAULT_PRIVACY: &str = "private";
const DEFAULT_WHISPER_BASE_URL: &str = "http://127.0.0.1:9000";
const DEFAULT_WHISPER_CONTAINER: &str = "youtube-assistant-whisper";
const DEFAULT_WHISPER_IMAGE: &str = "hwdsl2/whisper-server:latest";
const DEFAULT_WHISPER_MODEL: &str = "large-v3-turbo";
const STABILITY_POLLS: usize = 3;

type HttpsConnector =
    hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>;
type YouTubeHub = YouTube<HttpsConnector>;
type LlmClient = OpenAiClient<OpenAIConfig>;

#[derive(Debug, Parser)]
#[command(version, about = "Process videos and upload them to YouTube.")]
struct Args {
    /// Directory to monitor for video files.
    #[arg(long, env = "YOUTUBE_ASSISTANT_WATCH_DIR")]
    watch_dir: PathBuf,

    /// OAuth client secret JSON downloaded from Google Cloud Console. Not required with --no-upload.
    #[arg(long, env = "YOUTUBE_ASSISTANT_CLIENT_SECRET")]
    client_secret: Option<PathBuf>,

    /// Local file used to cache OAuth tokens after the first browser authorization.
    #[arg(
        long,
        env = "YOUTUBE_ASSISTANT_TOKEN_CACHE",
        default_value = ".youtube_token_cache.json"
    )]
    token_cache: PathBuf,

    /// OpenAI-compatible chat completions base URL, ending in /v1.
    #[arg(long, env = "YOUTUBE_ASSISTANT_OPENAI_BASE_URL")]
    openai_base_url: String,

    /// API key for the OpenAI-compatible metadata endpoint.
    #[arg(long, env = "YOUTUBE_ASSISTANT_OPENAI_API_KEY")]
    openai_api_key: String,

    /// Chat model to use for generating title and description.
    #[arg(long, env = "YOUTUBE_ASSISTANT_OPENAI_MODEL")]
    openai_model: String,

    /// YouTube video category id. 22 is People & Blogs.
    #[arg(
        long,
        env = "YOUTUBE_ASSISTANT_CATEGORY_ID",
        default_value = DEFAULT_CATEGORY_ID
    )]
    category_id: String,

    /// Privacy status for uploaded videos: private, unlisted, or public.
    #[arg(
        long,
        env = "YOUTUBE_ASSISTANT_PRIVACY",
        default_value = DEFAULT_PRIVACY
    )]
    privacy: String,

    /// Seconds a file's size must remain unchanged before processing starts.
    #[arg(long, env = "YOUTUBE_ASSISTANT_STABLE_SECONDS", default_value_t = 5)]
    stable_seconds: u64,

    /// Base URL for the local Whisper server.
    #[arg(
        long,
        env = "YOUTUBE_ASSISTANT_WHISPER_BASE_URL",
        default_value = DEFAULT_WHISPER_BASE_URL
    )]
    whisper_base_url: String,

    /// Whisper model name sent to the transcription endpoint.
    #[arg(
        long,
        env = "YOUTUBE_ASSISTANT_WHISPER_MODEL",
        default_value = DEFAULT_WHISPER_MODEL
    )]
    whisper_model: String,

    /// Optional Bearer token for Whisper servers started with WHISPER_API_KEY.
    #[arg(long, env = "YOUTUBE_ASSISTANT_WHISPER_API_KEY")]
    whisper_api_key: Option<String>,

    /// Docker image used when starting the local Whisper server.
    #[arg(
        long,
        env = "YOUTUBE_ASSISTANT_WHISPER_IMAGE",
        default_value = DEFAULT_WHISPER_IMAGE
    )]
    whisper_image: String,

    /// Docker container name used for the local Whisper server.
    #[arg(
        long,
        env = "YOUTUBE_ASSISTANT_WHISPER_CONTAINER",
        default_value = DEFAULT_WHISPER_CONTAINER
    )]
    whisper_container: String,

    /// Host port to bind to container port 9000 when managing Whisper with Docker.
    #[arg(long, env = "YOUTUBE_ASSISTANT_WHISPER_PORT", default_value_t = 9000)]
    whisper_port: u16,

    /// Run transcription and metadata generation but skip the actual YouTube upload.
    #[arg(long, env = "YOUTUBE_ASSISTANT_NO_UPLOAD")]
    no_upload: bool,
}

#[derive(Clone)]
struct AppClients {
    youtube: Option<YouTubeHub>,
    llm: LlmClient,
    http: reqwest::Client,
}

#[derive(Debug)]
struct ProcessingArtifacts {
    transcript: PathBuf,
    metadata: PathBuf,
    uploaded: PathBuf,
    upload_dry_run: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct VideoMetadata {
    title: String,
    description: String,
}

#[derive(Debug, Deserialize)]
struct WhisperTranscription {
    text: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct UploadMarker {
    video_id: String,
    youtube_url: String,
    uploaded_at_unix_seconds: u64,
    source_video: PathBuf,
    transcript_file: PathBuf,
    metadata_file: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct UploadDryRunMarker {
    skipped_upload: bool,
    would_upload_title: String,
    would_upload_description: String,
    recorded_at_unix_seconds: u64,
    source_video: PathBuf,
    transcript_file: PathBuf,
    metadata_file: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    info!(
        watch_dir = %args.watch_dir.display(),
        no_upload = args.no_upload,
        openai_base_url = %args.openai_base_url,
        openai_model = %args.openai_model,
        whisper_base_url = %args.whisper_base_url,
        "starting youtube assistant"
    );
    validate_args(&args)?;

    wait_for_whisper(&args).await?;

    let clients = AppClients {
        youtube: build_youtube_hub_if_needed(&args).await?,
        llm: build_llm_client(&args),
        http: reqwest::Client::new(),
    };

    run_processor(args, clients).await
}

fn init_tracing() {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("youtube_assistant=info,warn"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(ErrorLayer::default())
        .with(fmt::layer().with_target(false))
        .init();
}

fn validate_args(args: &Args) -> Result<()> {
    debug!("validating command line arguments");
    if !args.watch_dir.is_dir() {
        return Err(anyhow!(
            "watch directory does not exist or is not a directory: {}",
            args.watch_dir.display()
        ));
    }

    if !args.no_upload {
        let client_secret = args
            .client_secret
            .as_ref()
            .ok_or_else(|| anyhow!("--client-secret is required unless --no-upload is set"))?;

        if !client_secret.is_file() {
            return Err(anyhow!(
                "OAuth client secret file does not exist: {}",
                client_secret.display()
            ));
        }
    }

    if args.openai_base_url.trim().is_empty() {
        return Err(anyhow!("OpenAI-compatible base URL is required"));
    }

    if args.openai_model.trim().is_empty() {
        return Err(anyhow!("OpenAI-compatible model name is required"));
    }

    match args.privacy.as_str() {
        "private" | "unlisted" | "public" => Ok(()),
        other => Err(anyhow!(
            "unsupported privacy value {other:?}; use private, unlisted, or public"
        )),
    }
}

async fn wait_for_whisper(args: &Args) -> Result<()> {
    let url = format!("{}/health", args.whisper_base_url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    info!(url = %url, "checking Whisper server readiness");

    for attempt in 1..=10 {
        trace!(attempt, url = %url, "probing Whisper health endpoint");
        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                info!(base_url = %args.whisper_base_url, "Whisper server is ready");
                return Ok(());
            }
            Ok(response) => {
                debug!(
                    attempt,
                    status = %response.status(),
                    "Whisper health endpoint returned non-success status"
                );
            }
            Err(error) => {
                debug!(attempt, error = %error, "Whisper health probe failed");
            }
        }

        sleep(Duration::from_secs(2)).await;
    }

    Err(anyhow!(
        "Whisper server is not reachable at {url}\nStart it yourself before running youtube_assistant. Example:\n{}",
        whisper_docker_command(args)
    ))
}

fn whisper_docker_command(args: &Args) -> String {
    format!(
        "docker run --rm --name {} -p 127.0.0.1:{}:9000/tcp -e WHISPER_API_KEY=local-whisper-key {}",
        args.whisper_container, args.whisper_port, args.whisper_image
    )
}

async fn build_youtube_hub_if_needed(args: &Args) -> Result<Option<YouTubeHub>> {
    if args.no_upload {
        info!("skipping YouTube client setup because --no-upload is set");
        return Ok(None);
    }

    let client_secret = args
        .client_secret
        .as_ref()
        .ok_or_else(|| anyhow!("--client-secret is required unless --no-upload is set"))?;
    let secret = yup_oauth2::read_application_secret(client_secret)
        .await
        .with_context(|| {
            format!(
                "failed to read OAuth client secret from {}",
                client_secret.display()
            )
        })?;

    info!("initializing YouTube API client");
    let connector = hyper_rustls::HttpsConnectorBuilder::new()
        .with_native_roots()
        .context("failed to load native TLS roots")?
        .https_or_http()
        .enable_http2()
        .build();

    let client = hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
        .build(connector.clone());

    let auth_client =
        hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
            .build(connector);

    let auth = InstalledFlowAuthenticator::with_client(
        secret,
        InstalledFlowReturnMethod::HTTPRedirect,
        yup_oauth2::client::CustomHyperClientBuilder::from(auth_client),
    )
    .persist_tokens_to_disk(&args.token_cache)
    .build()
    .await
    .context("failed to initialize OAuth installed-app flow")?;

    Ok(Some(YouTube::new(client, auth)))
}

fn build_llm_client(args: &Args) -> LlmClient {
    info!(
        base_url = %args.openai_base_url,
        model = %args.openai_model,
        "initializing OpenAI-compatible metadata client"
    );
    let config = OpenAIConfig::new()
        .with_api_base(args.openai_base_url.clone())
        .with_api_key(args.openai_api_key.clone());

    OpenAiClient::with_config(config)
}

async fn run_processor(args: Args, clients: AppClients) -> Result<()> {
    info!(watch_dir = %args.watch_dir.display(), "starting processor");
    let mut in_progress = HashSet::new();
    scan_watch_dir(&args, &clients, &mut in_progress).await?;
    watch_for_videos(args, clients, in_progress).await
}

async fn scan_watch_dir(
    args: &Args,
    clients: &AppClients,
    in_progress: &mut HashSet<PathBuf>,
) -> Result<()> {
    info!(watch_dir = %args.watch_dir.display(), "scanning watch directory");
    let entries = std::fs::read_dir(&args.watch_dir).with_context(|| {
        format!(
            "failed to read watch directory {}",
            args.watch_dir.display()
        )
    })?;

    for entry in entries {
        let path = entry?.path();
        trace!(path = %path.display(), "saw directory entry during startup scan");
        if should_handle_path(&path) {
            process_video(args, clients, in_progress, path).await;
        }
    }

    Ok(())
}

async fn watch_for_videos(
    args: Args,
    clients: AppClients,
    mut in_progress: HashSet<PathBuf>,
) -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<notify::Result<Event>>(100);
    let mut watcher = notify::recommended_watcher(move |event| {
        let _ = tx.blocking_send(event);
    })
    .context("failed to create filesystem watcher")?;

    watcher
        .watch(&args.watch_dir, RecursiveMode::NonRecursive)
        .with_context(|| format!("failed to watch {}", args.watch_dir.display()))?;

    info!(watch_dir = %args.watch_dir.display(), "watching directory for videos");

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("shutdown requested");
                return Ok(());
            }
            maybe_event = rx.recv() => {
                let Some(event) = maybe_event else {
                    return Ok(());
                };

                let event = event.context("watcher failed")?;
                debug!(event = ?event, "filesystem event received");
                if !is_create_or_modify(&event.kind) {
                    trace!(kind = ?event.kind, "ignoring filesystem event");
                    continue;
                }

                for path in event.paths {
                    if should_handle_path(&path) {
                        process_video(&args, &clients, &mut in_progress, path).await;
                    }
                }
            }
        }
    }
}

async fn process_video(
    args: &Args,
    clients: &AppClients,
    in_progress: &mut HashSet<PathBuf>,
    path: PathBuf,
) {
    let result = process_video_inner(args, clients, in_progress, path).await;
    if let Err(error) = result {
        error!(error = ?error, "video processing failed");
    }
}

async fn process_video_inner(
    args: &Args,
    clients: &AppClients,
    in_progress: &mut HashSet<PathBuf>,
    path: PathBuf,
) -> Result<()> {
    let path = path
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", path.display()))?;
    info!(path = %path.display(), "processing video");

    if !in_progress.insert(path.clone()) {
        debug!(path = %path.display(), "video is already in progress; skipping duplicate event");
        return Ok(());
    }

    let result = process_video_pipeline(args, clients, &path).await;
    in_progress.remove(&path);
    result
}

async fn process_video_pipeline(args: &Args, clients: &AppClients, path: &Path) -> Result<()> {
    let artifacts = ProcessingArtifacts::for_video(path)?;
    debug!(
        path = %path.display(),
        transcript = %artifacts.transcript.display(),
        metadata = %artifacts.metadata.display(),
        uploaded = %artifacts.uploaded.display(),
        dry_run = %artifacts.upload_dry_run.display(),
        "resolved processing artifacts"
    );

    if artifacts.uploaded.exists() {
        info!(path = %path.display(), marker = %artifacts.uploaded.display(), "video already uploaded; marker exists");
        return Ok(());
    }

    wait_until_file_is_stable(
        path,
        Duration::from_secs(args.stable_seconds),
        STABILITY_POLLS,
    )
    .await?;

    let transcript = ensure_transcript(args, clients, path, &artifacts).await?;
    let metadata = ensure_metadata(args, clients, &transcript, &artifacts).await?;
    upload_video(args, clients, path, &artifacts, &metadata).await
}

impl ProcessingArtifacts {
    fn for_video(path: &Path) -> Result<Self> {
        Ok(Self {
            transcript: sidecar_path(path, "txt")?,
            metadata: sidecar_path(path, "metadata.json")?,
            uploaded: sidecar_path(path, "uploaded.json")?,
            upload_dry_run: sidecar_path(path, "upload-dry-run.json")?,
        })
    }
}

fn sidecar_path(path: &Path, suffix: &str) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow!("video path has no valid file stem: {}", path.display()))?;

    Ok(parent.join(format!("{stem}.{suffix}")))
}

async fn ensure_transcript(
    args: &Args,
    clients: &AppClients,
    path: &Path,
    artifacts: &ProcessingArtifacts,
) -> Result<String> {
    if artifacts.transcript.exists() {
        info!(
            path = %path.display(),
            transcript = %artifacts.transcript.display(),
            "using existing transcript"
        );
        return tokio::fs::read_to_string(&artifacts.transcript)
            .await
            .with_context(|| format!("failed to read {}", artifacts.transcript.display()));
    }

    info!(path = %path.display(), "transcribing video");
    let transcript = transcribe_video(args, clients, path).await?;
    atomic_write_string(&artifacts.transcript, &transcript).await?;
    info!(
        transcript = %artifacts.transcript.display(),
        bytes = transcript.len(),
        "saved transcript"
    );

    Ok(transcript)
}

async fn transcribe_video(args: &Args, clients: &AppClients, path: &Path) -> Result<String> {
    let url = format!(
        "{}/v1/audio/transcriptions",
        args.whisper_base_url.trim_end_matches('/')
    );
    debug!(
        path = %path.display(),
        url = %url,
        model = %args.whisper_model,
        has_api_key = args.whisper_api_key.as_deref().is_some_and(|value| !value.is_empty()),
        "calling Whisper transcription endpoint"
    );
    let part = Part::file(path)
        .await
        .with_context(|| format!("failed to open {} for transcription", path.display()))?;
    let form = Form::new()
        .part("file", part)
        .text("model", args.whisper_model.clone())
        .text("response_format", "json");

    let mut request = clients.http.post(url).multipart(form);
    if let Some(api_key) = args
        .whisper_api_key
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        request = request.bearer_auth(api_key);
    }

    let response = request
        .send()
        .await
        .context("failed to call Whisper transcription endpoint")?
        .error_for_status()
        .with_context(|| {
            "Whisper transcription endpoint returned an error. If it says Missing Authorization header or Invalid API key, pass --whisper-api-key or set YOUTUBE_ASSISTANT_WHISPER_API_KEY."
        })?;

    let transcription: WhisperTranscription = response
        .json()
        .await
        .context("failed to decode Whisper transcription response")?;
    debug!(
        chars = transcription.text.chars().count(),
        "received Whisper transcription"
    );

    Ok(transcription.text)
}

async fn ensure_metadata(
    args: &Args,
    clients: &AppClients,
    transcript: &str,
    artifacts: &ProcessingArtifacts,
) -> Result<VideoMetadata> {
    if artifacts.metadata.exists() {
        info!(
            metadata = %artifacts.metadata.display(),
            "using existing generated metadata"
        );
        let content = tokio::fs::read_to_string(&artifacts.metadata)
            .await
            .with_context(|| format!("failed to read {}", artifacts.metadata.display()))?;
        return serde_json::from_str(&content)
            .with_context(|| format!("failed to parse {}", artifacts.metadata.display()));
    }

    info!("generating YouTube metadata");
    let metadata = generate_metadata(args, clients, transcript).await?;
    let serialized = serde_json::to_string_pretty(&metadata)?;
    atomic_write_string(&artifacts.metadata, &serialized).await?;
    info!(
        metadata = %artifacts.metadata.display(),
        title = %metadata.title,
        "saved generated metadata"
    );

    Ok(metadata)
}

async fn generate_metadata(
    args: &Args,
    clients: &AppClients,
    transcript: &str,
) -> Result<VideoMetadata> {
    let system_prompt = r#"You generate metadata for YouTube uploads.
Return only valid JSON with exactly these keys:
{"title":"...","description":"..."}
The title should be compelling, accurate, and no more than 100 characters.
The description should summarize the video clearly and may include short bullet points if useful.
Do not invent facts that are not supported by the transcript."#;

    let user_prompt = format!(
        "Create a YouTube title and description for this transcript:\n\n{}",
        transcript
    );

    let request = CreateChatCompletionRequestArgs::default()
        .model(args.openai_model.clone())
        .max_completion_tokens(700_u32)
        .temperature(0.4_f32)
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system_prompt)
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_prompt)
                .build()?
                .into(),
        ])
        .build()?;

    debug!(
        model = %args.openai_model,
        transcript_chars = transcript.chars().count(),
        "calling OpenAI-compatible chat completion endpoint"
    );
    let response = clients
        .llm
        .chat()
        .create(request)
        .await
        .context("metadata generation request failed")?;

    let content = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
        .ok_or_else(|| anyhow!("metadata generation response did not include text"))?;

    debug!(
        response_chars = content.chars().count(),
        "received metadata model response"
    );
    parse_metadata_response(content)
}

fn parse_metadata_response(content: &str) -> Result<VideoMetadata> {
    trace!("parsing metadata model response as JSON");
    let cleaned = strip_json_fence(content.trim());
    let metadata: VideoMetadata =
        serde_json::from_str(cleaned).context("model did not return valid metadata JSON")?;

    if metadata.title.trim().is_empty() {
        return Err(anyhow!("generated title is empty"));
    }

    if metadata.description.trim().is_empty() {
        return Err(anyhow!("generated description is empty"));
    }

    Ok(VideoMetadata {
        title: metadata.title.trim().to_string(),
        description: metadata.description.trim().to_string(),
    })
}

fn strip_json_fence(content: &str) -> &str {
    content
        .strip_prefix("```json")
        .or_else(|| content.strip_prefix("```"))
        .and_then(|rest| rest.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(content)
}

async fn upload_video(
    args: &Args,
    clients: &AppClients,
    path: &Path,
    artifacts: &ProcessingArtifacts,
    metadata: &VideoMetadata,
) -> Result<()> {
    if args.no_upload {
        info!(path = %path.display(), "no-upload mode is enabled; recording dry-run marker");
        return record_upload_dry_run(path, artifacts, metadata).await;
    }

    let youtube = clients
        .youtube
        .as_ref()
        .ok_or_else(|| anyhow!("YouTube client was not initialized"))?;
    let video = google_youtube3::api::Video {
        snippet: Some(google_youtube3::api::VideoSnippet {
            title: Some(metadata.title.clone()),
            description: Some(metadata.description.clone()),
            category_id: Some(args.category_id.clone()),
            ..Default::default()
        }),
        status: Some(google_youtube3::api::VideoStatus {
            privacy_status: Some(args.privacy.clone()),
            self_declared_made_for_kids: Some(false),
            ..Default::default()
        }),
        ..Default::default()
    };

    info!(
        path = %path.display(),
        title = %metadata.title,
        privacy = %args.privacy,
        category_id = %args.category_id,
        "uploading video to YouTube"
    );

    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mime_type = video_mime_type(path);
    let (_, uploaded) = youtube
        .videos()
        .insert(video)
        .upload_resumable(file, mime_type)
        .await
        .with_context(|| format!("YouTube upload failed for {}", path.display()))?;

    let video_id = uploaded
        .id
        .ok_or_else(|| anyhow!("YouTube upload succeeded but response did not include video id"))?;
    info!(path = %path.display(), video_id = %video_id, "YouTube upload completed");
    let marker = UploadMarker {
        youtube_url: format!("https://www.youtube.com/watch?v={video_id}"),
        video_id,
        uploaded_at_unix_seconds: unix_timestamp_seconds()?,
        source_video: path.to_path_buf(),
        transcript_file: artifacts.transcript.clone(),
        metadata_file: artifacts.metadata.clone(),
    };
    let serialized = serde_json::to_string_pretty(&marker)?;
    atomic_write_string(&artifacts.uploaded, &serialized).await?;

    info!(
        path = %path.display(),
        youtube_url = %marker.youtube_url,
        marker = %artifacts.uploaded.display(),
        "saved upload marker"
    );

    Ok(())
}

async fn record_upload_dry_run(
    path: &Path,
    artifacts: &ProcessingArtifacts,
    metadata: &VideoMetadata,
) -> Result<()> {
    if artifacts.upload_dry_run.exists() {
        info!(
            path = %path.display(),
            marker = %artifacts.upload_dry_run.display(),
            "dry-run upload marker already exists; skipping"
        );
        return Ok(());
    }

    let marker = UploadDryRunMarker {
        skipped_upload: true,
        would_upload_title: metadata.title.clone(),
        would_upload_description: metadata.description.clone(),
        recorded_at_unix_seconds: unix_timestamp_seconds()?,
        source_video: path.to_path_buf(),
        transcript_file: artifacts.transcript.clone(),
        metadata_file: artifacts.metadata.clone(),
    };
    let serialized = serde_json::to_string_pretty(&marker)?;
    atomic_write_string(&artifacts.upload_dry_run, &serialized).await?;

    info!(
        path = %path.display(),
        marker = %artifacts.upload_dry_run.display(),
        "saved dry-run upload marker"
    );

    Ok(())
}

fn unix_timestamp_seconds() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_secs())
}

async fn atomic_write_string(path: &Path, content: &str) -> Result<()> {
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("tmp")
    ));

    trace!(path = %path.display(), tmp_path = %tmp_path.display(), bytes = content.len(), "writing file atomically");
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("failed to create {}", tmp_path.display()))?;
    tokio::io::AsyncWriteExt::write_all(&mut file, content.as_bytes())
        .await
        .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    tokio::io::AsyncWriteExt::flush(&mut file)
        .await
        .with_context(|| format!("failed to flush {}", tmp_path.display()))?;
    drop(file);

    tokio::fs::rename(&tmp_path, path)
        .await
        .with_context(|| format!("failed to move {} into place", path.display()))?;

    trace!(path = %path.display(), "atomic write complete");
    Ok(())
}

async fn wait_until_file_is_stable(
    path: &Path,
    interval: Duration,
    stable_polls: usize,
) -> Result<()> {
    let mut last_size = None;
    let mut unchanged = 0;
    debug!(
        path = %path.display(),
        interval_ms = interval.as_millis(),
        stable_polls,
        "waiting for file to become stable"
    );

    while unchanged < stable_polls {
        let size = tokio::fs::metadata(path)
            .await
            .with_context(|| format!("failed to stat {}", path.display()))?
            .len();

        if Some(size) == last_size {
            unchanged += 1;
        } else {
            trace!(path = %path.display(), size, "observed file size change");
            last_size = Some(size);
            unchanged = 0;
        }

        if unchanged < stable_polls {
            sleep(interval).await;
        }
    }

    debug!(path = %path.display(), "file is stable");
    Ok(())
}

fn should_handle_path(path: &Path) -> bool {
    path.is_file() && is_supported_video_extension(path)
}

fn is_supported_video_extension(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
        return false;
    };

    matches!(
        extension.to_ascii_lowercase().as_str(),
        "mp4" | "mov" | "m4v" | "mkv" | "webm" | "avi"
    )
}

fn video_mime_type(path: &Path) -> mime::Mime {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("mp4") | Some("m4v") => "video/mp4",
        Some("mov") => "video/quicktime",
        Some("mkv") => "video/x-matroska",
        Some("webm") => "video/webm",
        Some("avi") => "video/x-msvideo",
        _ => "application/octet-stream",
    }
    .parse()
    .expect("static MIME type is valid")
}

fn is_create_or_modify(kind: &EventKind) -> bool {
    matches!(kind, EventKind::Create(_) | EventKind::Modify(_))
}
