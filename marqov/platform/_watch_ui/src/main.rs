use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    prelude::*,
    widgets::{Block, BorderType, List, ListItem, ListState, Padding, Paragraph, Wrap},
};
use serde::{Deserialize, Serialize};
use std::{
    env, fs, io,
    io::Read,
    path::PathBuf,
    time::{Duration, Instant},
};
const MAX_BYTES: u64 = 1024 * 1024;
#[derive(Debug, Deserialize, Serialize)]
struct Snapshot {
    schema_version: u32,
    source: String,
    observed_at: String,
    freshness: String,
    jobs: Vec<Job>,
    #[serde(default)]
    project_name: Option<String>,
    #[serde(default)]
    message: Option<String>,
}
#[derive(Debug, Deserialize, Serialize)]
struct Job {
    workflow_id: String,
    provider_job_id: Option<String>,
    slurm_job_id: Option<String>,
    state: String,
    reason: String,
    provenance: String,
    cancellation: String,
}
fn load(path: &PathBuf) -> Result<Snapshot, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    if file.metadata()?.len() > MAX_BYTES {
        return Err("snapshot exceeds 1 MiB".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("snapshot exceeds 1 MiB".into());
    }
    let snapshot: Snapshot = serde_json::from_slice(&bytes)?;
    if snapshot.schema_version != 1 || snapshot.jobs.len() > 1000 {
        return Err("unsupported schema or more than 1000 jobs".into());
    }
    Ok(snapshot)
}
fn presentation(job: &Job) -> &str {
    if job.cancellation == "requested"
        && !matches!(job.state.as_str(), "completed" | "failed" | "cancelled")
    {
        return "Cancellation requested";
    }
    match job.state.as_str() {
        "queued" | "pending" | "scheduled" => "Queued",
        "running" => "Running",
        "needs_review" => "Needs review",
        "checking_execution" => "Checking execution",
        "cancelling" => "Cancellation requested",
        "cancelled" if job.cancellation == "confirmed" => "Cancelled",
        "completed" => "Completed",
        "failed" | "dispatch_failed" => "Failed",
        _ => "Needs review (unknown evidence)",
    }
}
const BACKGROUND: Color = Color::Rgb(15, 21, 32);
const FOREGROUND: Color = Color::Rgb(222, 231, 242);
const MUTED: Color = Color::Rgb(147, 163, 184);
const BORDER: Color = Color::Rgb(54, 70, 92);
const TEAL: Color = Color::Rgb(77, 219, 198);
const AMBER: Color = Color::Rgb(246, 194, 91);
const SELECTED: Color = Color::Rgb(27, 55, 68);
fn status_color(job: &Job) -> Color {
    match presentation(job) {
        "Running" => Color::Rgb(110, 175, 255),
        "Completed" => Color::Rgb(111, 213, 154),
        "Failed" => Color::Rgb(255, 126, 136),
        "Cancelled" => MUTED,
        _ => AMBER,
    }
}
fn panel(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER))
        .title(Span::styled(title, Style::default().fg(TEAL).bold()))
        .padding(Padding::horizontal(1))
}
struct App {
    snapshot: Snapshot,
    selected: usize,
    details: bool,
    error: Option<String>,
    refreshes: usize,
}
impl App {
    fn move_by(&mut self, delta: isize) {
        self.selected = self
            .selected
            .saturating_add_signed(delta)
            .min(self.snapshot.jobs.len().saturating_sub(1));
    }
    fn refresh(&mut self, path: &PathBuf) {
        self.refreshes += 1;
        match load(path) {
            Ok(snapshot) => {
                let identity = self
                    .snapshot
                    .jobs
                    .get(self.selected)
                    .map(|j| j.workflow_id.clone());
                self.selected = identity
                    .and_then(|id| snapshot.jobs.iter().position(|j| j.workflow_id == id))
                    .unwrap_or(self.selected);
                self.snapshot = snapshot;
                self.error = None;
                self.move_by(0);
            }
            Err(e) => {
                self.snapshot.freshness = "stale".into();
                self.error = Some(e.to_string());
            }
        }
    }
    fn render(&self, f: &mut Frame) {
        f.render_widget(
            Block::default().style(Style::default().bg(BACKGROUND).fg(FOREGROUND)),
            f.area(),
        );
        let areas = Layout::vertical([
            Constraint::Length(6),
            Constraint::Min(1),
            Constraint::Length(2),
        ])
        .split(f.area());
        let freshness = match self.snapshot.freshness.as_str() {
            "fresh" => "FRESH AT CAPTURE",
            "stale" => "STALE — observation may have changed",
            _ => "UNKNOWN freshness — do not infer current state",
        };
        let mut counts = [0usize; 6];
        for job in &self.snapshot.jobs {
            counts[match presentation(job) {
                "Running" => 0,
                "Queued" => 1,
                "Completed" => 3,
                "Failed" => 4,
                "Cancelled" => 5,
                _ => 2,
            }] += 1;
        }
        let labels = [
            "running",
            "queued",
            "attention",
            "completed",
            "failed",
            "cancelled",
        ];
        let colors = [
            Color::Rgb(110, 175, 255),
            AMBER,
            AMBER,
            Color::Rgb(111, 213, 154),
            Color::Rgb(255, 126, 136),
            MUTED,
        ];
        let mut summary = Vec::new();
        for i in 0..6 {
            if i > 0 {
                summary.push(Span::styled("  ·  ", Style::default().fg(BORDER)));
            }
            summary.push(Span::styled(
                format!("{} {}", counts[i], labels[i]),
                Style::default().fg(colors[i]),
            ));
        }
        let header = vec![
            Line::from(vec![
                Span::styled(
                    "MARQOV  /  WORKFLOW INSPECTOR",
                    Style::default().fg(TEAL).bold(),
                ),
                Span::styled(
                    format!("    {} · READ ONLY", self.snapshot.source.to_uppercase()),
                    Style::default().fg(MUTED),
                ),
            ]),
            Line::from(summary),
            Line::from(Span::styled(
                self.snapshot.message.as_deref().unwrap_or(freshness),
                Style::default().fg(if self.snapshot.freshness == "fresh" {
                    MUTED
                } else {
                    AMBER
                }),
            )),
            Line::from(Span::styled(
                format!(
                    "{} · observed {}",
                    self.snapshot
                        .project_name
                        .as_deref()
                        .unwrap_or("All team jobs"),
                    self.snapshot.observed_at
                ),
                Style::default().fg(MUTED),
            )),
        ];
        f.render_widget(Paragraph::new(header).block(panel(" Marqov ")), areas[0]);
        let wide = self.details && f.area().width >= 75;
        let columns = if wide {
            Layout::horizontal([Constraint::Percentage(48), Constraint::Percentage(52)])
                .split(areas[1])
        } else {
            Layout::horizontal([Constraint::Percentage(100)]).split(areas[1])
        };
        let rows: Vec<ListItem> = self
            .snapshot
            .jobs
            .iter()
            .map(|j| {
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled("● ", Style::default().fg(status_color(j))),
                        Span::styled(
                            format!("{}  ", j.workflow_id),
                            Style::default().fg(FOREGROUND).bold(),
                        ),
                        Span::styled(presentation(j), Style::default().fg(status_color(j))),
                    ]),
                    Line::from(Span::styled(
                        format!("  {}", j.reason),
                        Style::default().fg(MUTED),
                    )),
                ])
            })
            .collect();
        let mut state = ListState::default().with_selected(if rows.is_empty() {
            None
        } else {
            Some(self.selected)
        });
        f.render_stateful_widget(
            List::new(rows)
                .block(panel(" Workflow jobs "))
                .highlight_style(Style::default().bg(SELECTED))
                .highlight_symbol("▎ "),
            columns[0],
            &mut state,
        );
        if wide && let Some(j) = self.snapshot.jobs.get(self.selected) {
            let text = vec![
                Line::from(Span::styled(
                    &j.workflow_id,
                    Style::default().fg(FOREGROUND).bold(),
                )),
                Line::from(Span::styled(
                    presentation(j),
                    Style::default().fg(status_color(j)).bold(),
                )),
                Line::from(""),
                Line::from(format!(
                    "Provider: {}",
                    j.provider_job_id.as_deref().unwrap_or("unknown")
                )),
                Line::from(format!(
                    "Slurm: {}",
                    j.slurm_job_id.as_deref().unwrap_or("unknown")
                )),
                Line::from(format!("Workflow cancellation: {}", j.cancellation)),
                Line::from(""),
                Line::from(j.reason.as_str()),
                Line::from(""),
                Line::from(Span::styled(
                    "Provenance:",
                    Style::default().fg(TEAL).bold(),
                )),
                Line::from(Span::styled(&j.provenance, Style::default().fg(MUTED))),
                Line::from(""),
                Line::from(Span::styled(
                    "Read-only. Workflow state does not prove resources freed. Freshness describes snapshot fetch only. Results remain server-owned.",
                    Style::default().fg(MUTED),
                )),
            ];
            f.render_widget(
                Paragraph::new(text)
                    .wrap(Wrap { trim: true })
                    .block(panel(" Selected workflow ")),
                columns[1],
            );
        }
        let footer = self.error.as_deref().unwrap_or(if self.details && !wide {
            "↑↓ select · d details (needs width ≥75) · r refresh · q exit"
        } else {
            "↑↓ select · d details · r refresh · q exit"
        });
        f.render_widget(
            Paragraph::new(footer)
                .style(Style::default().fg(if self.error.is_some() { AMBER } else { MUTED }))
                .wrap(Wrap { trim: true }),
            areas[2],
        );
    }
}
struct TerminalGuard;
impl TerminalGuard {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        if let Err(e) = execute!(io::stdout(), EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(e);
        }
        Ok(Self)
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/journeys.json"
    ));
    let mut dump = false;
    let mut render_size: Option<(u16, u16)> = None;
    let mut render_json = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--snapshot" => path = args.next().ok_or("--snapshot requires a path")?.into(),
            "--dump" => dump = true,
            "--render" | "--render-json" => {
                render_json = arg == "--render-json";
                let width: u16 = args
                    .next()
                    .ok_or("--render requires width height")?
                    .parse()?;
                let height: u16 = args
                    .next()
                    .ok_or("--render requires width height")?
                    .parse()?;
                if width == 0 || height == 0 || width > 300 || height > 100 {
                    return Err("render bounds: width 1..300, height 1..100".into());
                }
                render_size = Some((width, height));
            }
            "--help" => {
                println!(
                    "marqov-ratatui-inspector [--snapshot PATH] [--dump]\nRead-only local snapshot; manual refresh limited to one per second."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let snapshot = load(&path)?;
    if dump {
        println!("{}", serde_json::to_string_pretty(&snapshot)?);
        return Ok(());
    }
    if let Some((width, height)) = render_size {
        let app = App {
            snapshot,
            selected: 0,
            details: true,
            error: None,
            refreshes: 0,
        };
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(width, height))?;
        terminal.draw(|f| app.render(f))?;
        if render_json {
            let cells: Vec<_> = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "text": c.symbol(), "fg": c.fg.to_string(), "bg": c.bg.to_string(),
                        "bold": c.modifier.contains(Modifier::BOLD)
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::json!({"width": width, "height": height, "cells": cells})
            );
            return Ok(());
        }
        for row in terminal.backend().buffer().content.chunks(width as usize) {
            println!("{}", row.iter().map(|c| c.symbol()).collect::<String>());
        }
        return Ok(());
    }
    let _guard = TerminalGuard::enter()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let mut app = App {
        snapshot,
        selected: 0,
        details: true,
        error: None,
        refreshes: 0,
    };
    let mut last_refresh = Instant::now() - Duration::from_secs(1);
    let mut last_auto_refresh = Instant::now();
    loop {
        if last_auto_refresh.elapsed() >= Duration::from_secs(1) {
            app.refresh(&path);
            last_auto_refresh = Instant::now();
        }
        terminal.draw(|frame| app.render(frame))?;
        if event::poll(Duration::from_millis(250))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                break;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Down | KeyCode::Char('j') => app.move_by(1),
                KeyCode::Up | KeyCode::Char('k') => app.move_by(-1),
                KeyCode::Char('d') => app.details = !app.details,
                KeyCode::Char('r') if last_refresh.elapsed() >= Duration::from_secs(1) => {
                    app.refresh(&path);
                    last_refresh = Instant::now();
                }
                _ => {}
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    fn app() -> App {
        App {
            snapshot: serde_json::from_str(include_str!("../fixtures/journeys.json")).unwrap(),
            selected: 0,
            details: true,
            error: None,
            refreshes: 0,
        }
    }
    fn screen(app: &App, width: u16, height: u16) -> String {
        let mut t = Terminal::new(TestBackend::new(width, height)).unwrap();
        t.draw(|f| app.render(f)).unwrap();
        t.backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect()
    }
    #[test]
    fn cancellation_identity_and_unknown_are_conservative() {
        let mut a = app();
        assert_eq!(presentation(&a.snapshot.jobs[4]), "Cancellation requested");
        assert_eq!(presentation(&a.snapshot.jobs[5]), "Cancelled");
        let j = &mut a.snapshot.jobs[2];
        assert!(j.provider_job_id.is_none());
        assert_eq!(j.slurm_job_id.as_deref(), Some("423"));
        j.state = "future_state".into();
        assert!(presentation(j).contains("unknown"));
        j.state = "cancelled".into();
        assert_ne!(presentation(j), "Cancelled");
    }
    #[test]
    fn details_and_resize_preserve_selection_without_io() {
        let mut a = app();
        a.move_by(3);
        let full = screen(&a, 120, 35);
        assert!(full.contains("Checking execution"));
        assert!(full.contains("Provider: unknown"));
        assert!(full.contains("Synthetic sanitized"));
        for (w, h) in [(40, 15), (10, 5), (1, 1), (120, 35)] {
            let _ = screen(&a, w, h);
            assert_eq!(a.selected, 3);
        }
        a.details = false;
        assert!(!screen(&a, 120, 35).contains("Provenance:"));
        assert_eq!(a.refreshes, 0);
    }
    #[test]
    fn failed_refresh_retains_observation_and_marks_stale() {
        let mut a = app();
        a.refresh(&PathBuf::from("/nonexistent-mrq-snapshot"));
        assert_eq!(a.snapshot.jobs.len(), 8);
        assert_eq!(a.snapshot.freshness, "stale");
        assert!(a.error.is_some());
        a.snapshot.freshness = "new_unknown".into();
        a.error = None;
        assert!(screen(&a, 120, 35).contains("UNKNOWN freshness"));
    }
    #[test]
    fn refresh_preserves_identity_after_reorder_and_enforces_size() {
        let mut a = app();
        a.selected = 3;
        let path =
            std::env::temp_dir().join(format!("ratatui-refresh-{}.json", std::process::id()));
        let mut reordered: Snapshot =
            serde_json::from_str(include_str!("../fixtures/journeys.json")).unwrap();
        reordered.jobs.reverse();
        fs::write(&path, serde_json::to_vec(&reordered).unwrap()).unwrap();
        a.refresh(&path);
        assert_eq!(a.snapshot.jobs[a.selected].workflow_id, "workflow-4");
        fs::write(&path, vec![b' '; MAX_BYTES as usize + 1]).unwrap();
        assert!(load(&path).unwrap_err().to_string().contains("1 MiB"));
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn status_colours_selection_and_summary_match_observed_states() {
        let a = app();
        let mut terminal = Terminal::new(TestBackend::new(120, 35)).unwrap();
        terminal.draw(|f| a.render(f)).unwrap();
        let buffer = terminal.backend().buffer();
        assert!(buffer.content.iter().any(|cell| cell.bg == SELECTED));
        assert!(
            buffer
                .content
                .iter()
                .any(|cell| cell.fg == status_color(&a.snapshot.jobs[7]))
        );
        let text = screen(&a, 120, 35);
        for label in [
            "1 running",
            "1 queued",
            "3 attention",
            "1 completed",
            "1 failed",
            "1 cancelled",
        ] {
            assert!(text.contains(label), "missing summary: {label}");
        }
        assert_eq!(status_color(&a.snapshot.jobs[4]), AMBER);
        assert_ne!(
            status_color(&a.snapshot.jobs[1]),
            status_color(&a.snapshot.jobs[7])
        );
    }
    #[test]
    fn selection_clamps_empty_and_edges() {
        let mut a = app();
        a.move_by(-9);
        assert_eq!(a.selected, 0);
        a.move_by(100);
        assert_eq!(a.selected, 7);
        a.snapshot.jobs.clear();
        a.move_by(0);
        assert_eq!(a.selected, 0);
        let _ = screen(&a, 80, 24);
    }
}
