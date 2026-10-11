use std::{
    hash::{Hash, Hasher},
    num::NonZeroU32,
    ops::Range,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

use collections::{FxHasher, HashMap, HashSet};
use gpui::{
    Font, FontId, FontStyle, FontWeight, GlyphId, HighlightStyle, Hsla, LineLayout, Pixels,
    StrikethroughStyle, TextRun, UnderlineStyle, WindowTextSystem,
};
use language::LanguageAwareStyling;
use parking_lot::Mutex;
use text::{Bias, Edit, Patch};

use unicode_segmentation::{GraphemeCursor, UnicodeSegmentation};
use util::debug_panic;

use crate::{
    EditorStyle, MAX_LINE_LEN,
    display_map::{
        ChunkRendererId, ChunkReplacement, DisplayPoint, DisplayRow, DisplaySnapshot,
        EditPredictionStyles, HighlightedChunk,
        long_rows::{
            DiagnosticKey, SyntaxStamp, TrailingWhitespaceRequest, diagnostic_keys,
            row_buffer_range,
        },
    },
    scroll::ScrollPixelOffset,
};
use project::{InlayId, project_settings::DiagnosticSeverity};

const CHUNK_LEN: u32 = 2_048;
const MAX_CHUNK_LEN: u32 = 3 * CHUNK_LEN;
const FORCED_BOUNDARY_SLACK: u32 = 512;
pub(super) const SHAPING_CONTEXT: u32 = 64;
const FORCED_BOUNDARY_CONTEXT: u32 = MAX_LINE_LEN as u32 / 2;
const MAX_SHAPED_LEN: u32 = 10_240;
pub(super) const MAX_IDLE_REFRESHES: u8 = 3;
pub(super) const MAX_SYNC_RULER_LEN: u32 = 16 * 1024;
const MAX_RENDERED_LEN: u32 = 16 * MAX_SHAPED_LEN;
const SYNC_MARGIN: u32 = SHAPING_CONTEXT / 2;
const SYNC_GROWTH: u32 = 4;
const SYNC_TOLERANCE: f32 = 0.05;
pub(super) const PREFIX_MARGIN: u32 = 2 * MAX_SHAPED_LEN;

const _: () = assert!(
    SHAPING_CONTEXT <= FORCED_BOUNDARY_CONTEXT
        && FORCED_BOUNDARY_CONTEXT < MAX_CHUNK_LEN
        && CHUNK_LEN
            + MAX_CHUNK_LEN
            + FORCED_BOUNDARY_SLACK
            + 4
            + 2 * (FORCED_BOUNDARY_CONTEXT + char::MAX_LEN_UTF8 as u32)
            <= MAX_SHAPED_LEN
);

#[derive(Clone)]
pub(crate) struct RulerShaper {
    pub(crate) text_system: Arc<WindowTextSystem>,
    pub(crate) style: EditorStyle,
    pub(crate) font_size: Pixels,
    pub(crate) language_aware: LanguageAwareStyling,
}

impl RulerShaper {
    pub(crate) fn layout_columns(
        &self,
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        columns: Range<u32>,
    ) -> Arc<LineLayout> {
        let mut text = String::new();
        let mut runs = Vec::new();
        for chunk in self.chunks(snapshot, row, columns) {
            text.push_str(chunk.text);
            runs.push(self.run(chunk.style, chunk.text.len()));
        }
        self.layout(&text, &runs)
    }

    pub(crate) fn compose_columns(
        &self,
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        columns: Range<u32>,
    ) -> ComposedLayout {
        let mut composed = ComposedLayout::default();
        let mut text = String::new();
        let mut runs = Vec::new();
        for chunk in self.chunks(snapshot, row, columns) {
            let Some(replacement) = &chunk.replacement else {
                text.push_str(chunk.text);
                runs.push(self.run(chunk.style, chunk.text.len()));
                continue;
            };
            if !text.is_empty() {
                composed.push_text(self.layout(&text, &runs));
                text.clear();
                runs.clear();
            }
            let width = match replacement {
                ChunkReplacement::Str(replacement) => {
                    self.layout(replacement, &[self.run(chunk.style, replacement.len())])
                        .width
                }
                ChunkReplacement::Renderer(renderer) => {
                    renderer.measured_width.unwrap_or_else(|| {
                        self.layout(chunk.text, &[self.run(chunk.style, chunk.text.len())])
                            .width
                    })
                }
            };
            composed.push_replacement(chunk.text.len(), width);
        }
        if !text.is_empty() {
            composed.push_text(self.layout(&text, &runs));
        }
        composed
    }

    fn chunks<'a>(
        &'a self,
        snapshot: &'a DisplaySnapshot,
        row: DisplayRow,
        columns: Range<u32>,
    ) -> impl Iterator<Item = HighlightedChunk<'a>> + 'a {
        snapshot.highlighted_chunks_in_range(
            DisplayPoint::new(row, columns.start)..DisplayPoint::new(row, columns.end),
            self.language_aware,
            &self.style,
        )
    }

    fn run(&self, style: Option<HighlightStyle>, len: usize) -> TextRun {
        match style {
            Some(style) => self.style.text.clone().highlight(style).to_run(len),
            None => self.style.text.to_run(len),
        }
    }

    fn layout(&self, text: &str, runs: &[TextRun]) -> Arc<LineLayout> {
        self.text_system
            .layout_line(text, self.font_size, runs, None)
    }

    fn layout_uncached(&self, text: &str, runs: &[TextRun]) -> Arc<LineLayout> {
        Arc::new(
            self.text_system
                .layout_line_uncached(text, self.font_size, runs),
        )
    }

    fn uncached_width(&self, text: &str, style: Option<HighlightStyle>) -> Pixels {
        self.layout_uncached(text, &[self.run(style, text.len())])
            .width
    }

    fn renderer_metrics_key(&self) -> u64 {
        renderer_metrics_key(&self.style.text.font(), self.font_size)
    }

    pub(super) fn metrics_fingerprint(&self) -> u64 {
        let EditorStyle {
            background,
            border: _,
            local_player: _,
            text,
            scrollbar_width: _,
            syntax,
            status,
            inlay_hints_style,
            edit_prediction_styles:
                EditPredictionStyles {
                    insertion,
                    whitespace,
                },
            unnecessary_code_fade,
            show_underlines,
        } = &self.style;
        let mut hasher = FxHasher::default();
        text.font().hash(&mut hasher);
        hasher.write_u32(f32::from(self.font_size).to_bits());
        text.color.hash(&mut hasher);
        text.underline.hash(&mut hasher);
        text.strikethrough.hash(&mut hasher);
        background.hash(&mut hasher);
        for color in [
            status.error,
            status.warning,
            status.info,
            status.hint,
            status.ignored,
        ] {
            color.hash(&mut hasher);
        }
        hasher.write_u64(syntax.fingerprint());
        for style in [inlay_hints_style, insertion, whitespace] {
            HighlightStyle {
                background_color: None,
                ..*style
            }
            .hash(&mut hasher);
        }
        hasher.write_u8(self.language_aware.tree_sitter as u8);
        hasher.write_u8(self.language_aware.diagnostics as u8);
        hasher.write_u8(*show_underlines as u8);
        hasher.write_u32(unnecessary_code_fade.to_bits());
        hasher.finish()
    }

    fn matches(&self, ruler: &RowRuler) -> bool {
        ruler.metrics_fingerprint == self.metrics_fingerprint()
    }
}

pub(crate) fn renderer_metrics_key(font: &Font, font_size: Pixels) -> u64 {
    let mut hasher = FxHasher::default();
    font.hash(&mut hasher);
    hasher.write_u32(f32::from(font_size).to_bits());
    hasher.finish()
}

#[derive(Default)]
pub(crate) struct ComposedLayout {
    pieces: Vec<ComposedPiece>,
    len: usize,
    width: Pixels,
}

struct ComposedPiece {
    start: usize,
    len: usize,
    x: Pixels,
    kind: PieceKind,
}

enum PieceKind {
    Text(Arc<LineLayout>),
    Replacement(Pixels),
}

impl ComposedLayout {
    pub(crate) fn from_line(layout: Arc<LineLayout>) -> Self {
        let mut composed = Self::default();
        composed.push_text(layout);
        composed
    }

    pub(crate) fn width(&self) -> Pixels {
        self.width
    }

    pub(crate) fn x_for_index(&self, index: usize) -> Pixels {
        let Some(piece) = self.piece_at(index) else {
            return Pixels::ZERO;
        };
        match &piece.kind {
            PieceKind::Text(layout) => piece.x + layout.x_for_index(index - piece.start),
            PieceKind::Replacement(width) if index > piece.start => piece.x + *width,
            PieceKind::Replacement(_) => piece.x,
        }
    }

    pub(crate) fn closest_index_for_x(&self, x: Pixels) -> usize {
        let ix = self
            .pieces
            .partition_point(|piece| piece.x <= x)
            .saturating_sub(1);
        let Some(piece) = self.pieces.get(ix) else {
            return 0;
        };
        match &piece.kind {
            PieceKind::Text(layout) => piece.start + layout.closest_index_for_x(x - piece.x),
            PieceKind::Replacement(width) if x - piece.x > *width / 2. => piece.start + piece.len,
            PieceKind::Replacement(_) => piece.start,
        }
    }

    fn is_glyph_boundary(&self, index: usize) -> bool {
        if index >= self.len {
            return true;
        }
        self.piece_at(index).is_none_or(|piece| match &piece.kind {
            PieceKind::Text(layout) => layout.is_glyph_boundary(index - piece.start),
            PieceKind::Replacement(_) => index == piece.start,
        })
    }

    fn glyphs_in(&self, range: Range<usize>) -> Vec<(Option<(FontId, GlyphId)>, usize, Pixels)> {
        let mut glyphs = Vec::new();
        for piece in &self.pieces {
            if piece.start + piece.len <= range.start || range.end <= piece.start {
                continue;
            }
            match &piece.kind {
                PieceKind::Text(layout) => {
                    for run in &layout.runs {
                        for glyph in &run.glyphs {
                            let index = piece.start + glyph.index;
                            if range.contains(&index) {
                                glyphs.push((
                                    Some((run.font_id, glyph.id)),
                                    index,
                                    piece.x + glyph.position.x,
                                ));
                            }
                        }
                    }
                }
                PieceKind::Replacement(_) => {
                    if range.contains(&piece.start) {
                        glyphs.push((None, piece.start, piece.x));
                    }
                }
            }
        }
        glyphs
    }

    fn piece_at(&self, index: usize) -> Option<&ComposedPiece> {
        let ix = self
            .pieces
            .partition_point(|piece| piece.start <= index)
            .saturating_sub(1);
        self.pieces.get(ix)
    }

    fn push_text(&mut self, layout: Arc<LineLayout>) {
        let (len, width) = (layout.len, layout.width);
        self.push(len, width, PieceKind::Text(layout));
    }

    fn push_replacement(&mut self, len: usize, width: Pixels) {
        self.push(len, width, PieceKind::Replacement(width));
    }

    fn push(&mut self, len: usize, width: Pixels, kind: PieceKind) {
        self.pieces.push(ComposedPiece {
            start: self.len,
            len,
            x: self.width,
            kind,
        });
        self.len += len;
        self.width += width;
    }
}

#[derive(Default)]
struct ReplacedChars(Vec<u64>);

impl ReplacedChars {
    fn insert(&mut self, offset: u32) {
        let word = offset as usize / 64;
        if self.0.len() <= word {
            self.0.resize(word + 1, 0);
        }
        self.0[word] |= 1 << (offset % 64);
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn contains(&self, offset: u32) -> bool {
        self.0
            .get(offset as usize / 64)
            .is_some_and(|word| word & (1 << (offset % 64)) != 0)
    }

    fn in_range(&self, range: Range<u32>) -> impl Iterator<Item = u32> + '_ {
        let end = (range.end as usize).min(self.0.len() * 64);
        let mut offset = range.start as usize;
        std::iter::from_fn(move || {
            while offset < end {
                let word = self.0[offset / 64] >> (offset % 64);
                if word == 0 {
                    offset = (offset / 64 + 1) * 64;
                    continue;
                }
                let found = offset + word.trailing_zeros() as usize;
                if found >= end {
                    offset = end;
                    return None;
                }
                offset = found + 1;
                return Some(found as u32);
            }
            None
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct FontOverrides {
    weight: Option<FontWeight>,
    style: Option<FontStyle>,
}

impl FontOverrides {
    fn from_style(style: Option<HighlightStyle>) -> Self {
        Self {
            weight: style.and_then(|style| style.font_weight),
            style: style.and_then(|style| style.font_style),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
struct Decorations {
    color: Hsla,
    underline: Option<UnderlineStyle>,
    strikethrough: Option<StrikethroughStyle>,
}

impl Decorations {
    fn from_run(run: &TextRun) -> Self {
        Self {
            color: run.color,
            underline: run.underline,
            strikethrough: run.strikethrough,
        }
    }
}

#[derive(Clone, Debug)]
struct StyleSegment {
    end: u32,
    overrides: FontOverrides,
    decorations: Decorations,
    style: Option<HighlightStyle>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RulerCacheVersion {
    pub(crate) tabs: usize,
    pub(crate) highlights: usize,
    pub(crate) non_text_state: usize,
    pub(crate) diagnostics_max_severity: DiagnosticSeverity,
    pub(crate) tab_size: NonZeroU32,
    pub(crate) masked: bool,
}

pub(crate) struct RowRulerCache {
    version: RulerCacheVersion,
    rulers: Mutex<HashMap<u32, CachedRuler>>,
    prefixes: Mutex<HashMap<u32, Arc<RowRuler>>>,
    previous: Mutex<HashMap<u32, PreviousRuler>>,
    renderer_widths: Mutex<(u64, HashMap<InlayId, Pixels>)>,
    trailing_whitespace: Mutex<HashMap<u32, TrailingWhitespace>>,
    grid_rejections: Mutex<HashSet<u64>>,
    uniform_ascii_fonts: Mutex<HashMap<u64, bool>>,
    requests: Weak<RulerRequests>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TrailingWhitespace {
    pub(crate) row_len: u32,
    pub(crate) start: usize,
    pub(crate) exact: bool,
    pub(crate) idle_refreshes: u8,
}

#[derive(Default)]
pub(crate) struct RulerRequests {
    rulers: Mutex<HashMap<u32, RulerRequest>>,
    trailing_whitespace: Mutex<HashMap<u32, TrailingWhitespaceRequest>>,
}

pub(crate) struct RulerRequest {
    snapshot: Arc<DisplaySnapshot>,
    row: DisplayRow,
    shaper: RulerShaper,
    previous: Option<Arc<RowRuler>>,
    measure_previous: bool,
}

pub(crate) enum MappedRow {
    Unchanged(u32),
    Edited {
        row: u32,
        len: u32,
        edits: Vec<Vec<RowEdit>>,
    },
    Changed(u32),
}

#[derive(Clone)]
pub(crate) struct PreviousRuler {
    pub(crate) ruler: Arc<RowRuler>,
    pub(crate) matches_text: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct RowEdit {
    pub(crate) old: Range<u32>,
    pub(crate) new_len: u32,
}

struct CachedRuler {
    ruler: Arc<RowRuler>,
    idle_refreshes: u8,
    verified: bool,
}

impl RowRulerCache {
    pub(crate) fn new(
        version: RulerCacheVersion,
        previous: Option<&RowRulerCache>,
        requests: &Arc<RulerRequests>,
        mut remap: impl FnMut(u32, &RowRuler, bool) -> (MappedRow, bool),
        mut remap_trailing_whitespace: impl FnMut(
            u32,
            TrailingWhitespace,
        ) -> Option<(u32, TrailingWhitespace)>,
    ) -> Self {
        let grid_rejections = previous
            .map(|previous| previous.grid_rejections.lock().clone())
            .unwrap_or_default();
        let uniform_ascii_fonts = previous
            .map(|previous| previous.uniform_ascii_fonts.lock().clone())
            .unwrap_or_default();
        let trailing_whitespace = previous
            .map(|previous| {
                previous
                    .trailing_whitespace
                    .lock()
                    .iter()
                    .filter(|(_, trailing)| trailing.idle_refreshes < MAX_IDLE_REFRESHES)
                    .filter_map(|(row, trailing)| {
                        let (row, trailing) = remap_trailing_whitespace(*row, *trailing)?;
                        Some((
                            row,
                            TrailingWhitespace {
                                idle_refreshes: trailing.idle_refreshes + 1,
                                ..trailing
                            },
                        ))
                    })
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();
        let (previous_rulers, retained, prefixes, renderer_widths) = previous
            .map(|previous| {
                let rulers = previous
                    .rulers
                    .lock()
                    .iter()
                    .map(|(row, cached)| {
                        (
                            *row,
                            cached.ruler.clone(),
                            cached.idle_refreshes,
                            cached.verified,
                        )
                    })
                    .collect::<Vec<_>>();
                let mut retained = HashMap::default();
                let mut reusable = HashMap::default();
                for (row, ruler, idle_refreshes, verified) in rulers.iter() {
                    let (mapped, current) =
                        remap(*row, ruler, *idle_refreshes < MAX_IDLE_REFRESHES);
                    let (row, remapped) = mapped.apply(ruler);
                    let Some(remapped) = remapped else {
                        reusable
                            .entry(row)
                            .or_insert_with(|| PreviousRuler::stale(ruler.clone()));
                        continue;
                    };
                    if current {
                        retained.insert(
                            row,
                            CachedRuler {
                                ruler: remapped.clone(),
                                idle_refreshes: idle_refreshes + 1,
                                verified: *verified,
                            },
                        );
                    }
                    reusable.insert(row, PreviousRuler::current(remapped));
                }
                let mut prefixes = HashMap::default();
                for (row, prefix) in previous.prefixes.lock().iter() {
                    let (mapped, current) = remap(*row, prefix, true);
                    let (row, remapped) = mapped.apply(prefix);
                    let Some(remapped) = remapped else {
                        continue;
                    };
                    if current {
                        prefixes.insert(row, remapped.clone());
                    }
                    reusable
                        .entry(row)
                        .or_insert_with(|| PreviousRuler::stale(remapped));
                }
                if rulers.is_empty() {
                    for (row, previous) in previous.previous.lock().iter() {
                        let ruler = &previous.ruler;
                        let (row, remapped) = remap(*row, ruler, false).0.apply(ruler);
                        match remapped {
                            Some(remapped) => {
                                reusable.insert(
                                    row,
                                    PreviousRuler {
                                        ruler: remapped,
                                        matches_text: previous.matches_text,
                                    },
                                );
                            }
                            None => {
                                reusable
                                    .entry(row)
                                    .or_insert_with(|| PreviousRuler::stale(ruler.clone()));
                            }
                        }
                    }
                }
                (
                    reusable,
                    retained,
                    prefixes,
                    previous.renderer_widths.lock().clone(),
                )
            })
            .unwrap_or_default();
        Self {
            version,
            rulers: Mutex::new(retained),
            prefixes: Mutex::new(prefixes),
            previous: Mutex::new(previous_rulers),
            renderer_widths: Mutex::new(renderer_widths),
            trailing_whitespace: Mutex::new(trailing_whitespace),
            grid_rejections: Mutex::new(grid_rejections),
            uniform_ascii_fonts: Mutex::new(uniform_ascii_fonts),
            requests: Arc::downgrade(requests),
        }
    }

    pub(crate) fn version(&self) -> RulerCacheVersion {
        self.version
    }

    pub(crate) fn update_renderer_widths(
        &self,
        widths: impl IntoIterator<Item = (ChunkRendererId, Pixels)>,
        metrics_key: u64,
    ) -> bool {
        let mut renderer_widths = self.renderer_widths.lock();
        let mut changed = false;
        if renderer_widths.0 != metrics_key {
            changed = !renderer_widths.1.is_empty();
            *renderer_widths = (metrics_key, HashMap::default());
        }
        let renderer_widths = &mut renderer_widths.1;
        for (id, width) in widths {
            if let ChunkRendererId::Inlay(id) = id
                && renderer_widths.insert(id, width) != Some(width)
            {
                changed = true;
            }
        }
        if changed {
            self.prefixes.lock().clear();
            let mut rulers = self.rulers.lock();
            if !rulers.is_empty() {
                *self.previous.lock() = std::mem::take(&mut *rulers)
                    .into_iter()
                    .map(|(row, cached)| (row, PreviousRuler::current(cached.ruler)))
                    .collect();
            }
        }
        changed
    }

    pub(crate) fn trailing_whitespace_start(
        &self,
        tab_row: u32,
        row_len: u32,
        compute: impl FnOnce() -> TrailingWhitespace,
    ) -> TrailingWhitespace {
        {
            let mut trailing_whitespace = self.trailing_whitespace.lock();
            if let Some(trailing) = trailing_whitespace.get_mut(&tab_row)
                && trailing.row_len == row_len
            {
                trailing.idle_refreshes = 0;
                return *trailing;
            }
        }
        let trailing = compute();
        self.set_trailing_whitespace(tab_row, trailing);
        trailing
    }

    pub(crate) fn set_trailing_whitespace(&self, tab_row: u32, trailing: TrailingWhitespace) {
        self.trailing_whitespace.lock().insert(tab_row, trailing);
    }

    pub(super) fn request_trailing_whitespace(
        &self,
        tab_row: u32,
        request: impl FnOnce() -> TrailingWhitespaceRequest,
    ) {
        if let Some(requests) = self.requests.upgrade() {
            requests
                .trailing_whitespace
                .lock()
                .entry(tab_row)
                .or_insert_with(request);
        }
    }

    pub(crate) fn is_grid_rejected(&self, metrics_key: u64) -> bool {
        self.grid_rejections.lock().contains(&metrics_key)
    }

    pub(crate) fn reject_grid(&self, metrics_key: u64) {
        self.grid_rejections.lock().insert(metrics_key);
    }

    pub(crate) fn has_uniform_ascii(
        &self,
        metrics_key: u64,
        measure: impl FnOnce() -> bool,
    ) -> bool {
        if let Some(uniform) = self.uniform_ascii_fonts.lock().get(&metrics_key) {
            return *uniform;
        }
        let uniform = measure();
        self.uniform_ascii_fonts.lock().insert(metrics_key, uniform);
        uniform
    }

    pub(crate) fn remove_renderer_widths(&self, inlays: &[InlayId]) {
        let mut renderer_widths = self.renderer_widths.lock();
        for inlay in inlays {
            renderer_widths.1.remove(inlay);
        }
    }

    pub(crate) fn current(
        &self,
        tab_row: u32,
        shaper: &RulerShaper,
    ) -> Option<(Arc<RowRuler>, bool)> {
        let mut rulers = self.rulers.lock();
        let cached = rulers
            .get_mut(&tab_row)
            .filter(|cached| shaper.matches(&cached.ruler))?;
        cached.idle_refreshes = 0;
        Some((cached.ruler.clone(), cached.verified))
    }

    pub(crate) fn set_current(&self, tab_row: u32, ruler: Arc<RowRuler>) {
        self.rulers.lock().insert(
            tab_row,
            CachedRuler {
                ruler,
                idle_refreshes: 0,
                verified: true,
            },
        );
    }

    pub(crate) fn insert_built(&self, tab_row: u32, ruler: Arc<RowRuler>) -> bool {
        let mut rulers = self.rulers.lock();
        if let Some(cached) = rulers.get(&tab_row)
            && !cached.ruler.needs_measuring()
        {
            if cached.ruler.metrics_fingerprint != ruler.metrics_fingerprint {
                self.previous
                    .lock()
                    .insert(tab_row, PreviousRuler::current(cached.ruler.clone()));
            } else if cached.verified {
                return false;
            }
        }
        rulers.insert(
            tab_row,
            CachedRuler {
                ruler,
                idle_refreshes: 0,
                verified: false,
            },
        );
        true
    }

    pub(crate) fn retire(&self, tab_row: u32) {
        if let Some(cached) = self.rulers.lock().remove(&tab_row) {
            self.previous
                .lock()
                .insert(tab_row, PreviousRuler::current(cached.ruler));
        }
    }

    pub(crate) fn retire_unmeasurable(&self, tab_row: u32, ruler: Arc<RowRuler>) {
        self.rulers.lock().remove(&tab_row);
        self.previous
            .lock()
            .insert(tab_row, PreviousRuler::stale(ruler));
    }

    pub(crate) fn previous(&self, tab_row: u32, shaper: &RulerShaper) -> Option<PreviousRuler> {
        self.previous
            .lock()
            .get(&tab_row)
            .filter(|previous| shaper.matches(&previous.ruler))
            .cloned()
            .or_else(|| self.prefix(tab_row, shaper).map(PreviousRuler::stale))
    }

    pub(crate) fn prefix(&self, tab_row: u32, shaper: &RulerShaper) -> Option<Arc<RowRuler>> {
        self.prefixes
            .lock()
            .get(&tab_row)
            .filter(|prefix| shaper.matches(prefix))
            .cloned()
    }

    pub(crate) fn set_prefix(&self, tab_row: u32, prefix: Arc<RowRuler>) {
        self.prefixes.lock().insert(tab_row, prefix);
    }

    pub(crate) fn set_previous(&self, tab_row: u32, ruler: Arc<RowRuler>) {
        self.previous
            .lock()
            .insert(tab_row, PreviousRuler::current(ruler));
    }

    pub(crate) fn renderer_widths(&self, shaper: &RulerShaper) -> HashMap<InlayId, Pixels> {
        let renderer_widths = self.renderer_widths.lock();
        if renderer_widths.0 == shaper.renderer_metrics_key() {
            renderer_widths.1.clone()
        } else {
            HashMap::default()
        }
    }

    pub(crate) fn request(&self, tab_row: u32, request: impl FnOnce() -> RulerRequest) {
        if let Some(requests) = self.requests.upgrade() {
            requests
                .rulers
                .lock()
                .entry(tab_row)
                .or_insert_with(request);
        }
    }
}

impl RulerRequests {
    pub(crate) fn take(&self) -> Vec<(u32, RulerRequest)> {
        self.rulers.lock().drain().collect()
    }

    pub(crate) fn clear(&self) {
        self.rulers.lock().clear();
    }

    pub(super) fn take_trailing_whitespace(&self) -> Vec<(u32, TrailingWhitespaceRequest)> {
        self.trailing_whitespace.lock().drain().collect()
    }
}

impl RulerRequest {
    pub(crate) fn new(
        snapshot: Arc<DisplaySnapshot>,
        row: DisplayRow,
        shaper: RulerShaper,
        previous: Option<Arc<RowRuler>>,
    ) -> Self {
        Self {
            snapshot,
            row,
            shaper,
            previous,
            measure_previous: false,
        }
    }

    pub(crate) fn measure(
        snapshot: Arc<DisplaySnapshot>,
        row: DisplayRow,
        shaper: RulerShaper,
        ruler: Arc<RowRuler>,
    ) -> Self {
        Self {
            measure_previous: true,
            ..Self::new(snapshot, row, shaper, Some(ruler))
        }
    }

    pub(crate) fn row(&self) -> DisplayRow {
        self.row
    }

    pub(crate) fn row_len(&self) -> u32 {
        self.snapshot.line_len(self.row)
    }

    pub(crate) fn shaper(&self) -> &RulerShaper {
        &self.shaper
    }

    pub(crate) fn build(
        self,
        renderer_widths: &HashMap<InlayId, Pixels>,
        cancelled: &AtomicBool,
    ) -> Option<RowRuler> {
        if self.measure_previous
            && let Some(measured) = self.previous.as_ref().and_then(|previous| {
                previous.measured_cancellable(
                    &self.snapshot,
                    self.row,
                    &self.shaper,
                    renderer_widths,
                    u32::MAX,
                    cancelled,
                )
            })
        {
            return Some(measured);
        }
        if cancelled.load(Ordering::Relaxed) {
            return None;
        }
        RowRuler::new_cancellable(
            &self.snapshot,
            self.row,
            &self.shaper,
            self.previous.as_deref(),
            renderer_widths,
            cancelled,
        )
    }
}

impl MappedRow {
    fn apply(self, ruler: &Arc<RowRuler>) -> (u32, Option<Arc<RowRuler>>) {
        match self {
            Self::Unchanged(row) => (row, Some(ruler.clone())),
            Self::Edited { row, edits, .. } => (row, ruler.interpolate(&edits).map(Arc::new)),
            Self::Changed(row) => (row, None),
        }
    }
}

impl PreviousRuler {
    fn current(ruler: Arc<RowRuler>) -> Self {
        Self {
            ruler,
            matches_text: true,
        }
    }

    fn stale(ruler: Arc<RowRuler>) -> Self {
        Self {
            ruler,
            matches_text: false,
        }
    }
}

impl RowEdit {
    pub(crate) fn delta(&self) -> i64 {
        i64::from(self.new_len) - i64::from(self.old.end - self.old.start)
    }
}

pub(crate) fn compose_row_edits<'a>(
    patches: impl IntoIterator<Item = &'a [RowEdit]>,
) -> Option<Vec<RowEdit>> {
    let mut composed = Patch::<u32>::empty();
    for edits in patches {
        let mut delta = 0i64;
        let mut patch = Vec::with_capacity(edits.len());
        for edit in edits {
            if edit.old.start > edit.old.end {
                return None;
            }
            let new_start = u32::try_from(i64::from(edit.old.start) + delta).ok()?;
            patch.push(Edit {
                old: edit.old.clone(),
                new: new_start..new_start.checked_add(edit.new_len)?,
            });
            delta += edit.delta();
        }
        composed = composed.compose(patch);
    }
    Some(
        composed
            .into_inner()
            .into_iter()
            .map(|edit| RowEdit {
                new_len: edit.new.end - edit.new.start,
                old: edit.old,
            })
            .collect(),
    )
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderPiece {
    pub(crate) chunk: Range<u32>,
    pub(crate) context: Range<u32>,
    pub(crate) width: Pixels,
    pub(crate) x: ScrollPixelOffset,
    pub(crate) shaped: bool,
}

#[derive(Clone, Debug)]
struct RulerChunk {
    len: u32,
    width: Pixels,
    hash: u64,
    fixed: bool,
    context_before: u32,
    context_after: u32,
    measured: bool,
    seam_after: bool,
}

impl RulerChunk {
    fn unmeasured(len: u32) -> Self {
        Self {
            len,
            width: Pixels::ZERO,
            hash: 0,
            fixed: false,
            context_before: 0,
            context_after: 0,
            measured: false,
            seam_after: false,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RowRuler {
    metrics_fingerprint: u64,
    syntax: Option<SyntaxStamp>,
    diagnostics: Vec<DiagnosticKey>,
    chunks: Vec<RulerChunk>,
    starts: Vec<u32>,
    xs: Vec<ScrollPixelOffset>,
}

struct RowText {
    text: String,
    fixed_spans: Vec<(Range<u32>, Pixels)>,
    replaced: ReplacedChars,
    replacement_widths: HashMap<(char, FontOverrides), Pixels>,
    segments: Vec<StyleSegment>,
    forced_boundaries: Vec<u32>,
    hard_breaks: Vec<u32>,
}

impl RowText {
    fn read(
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        shaper: &RulerShaper,
        renderer_widths: &HashMap<InlayId, Pixels>,
        columns: Range<u32>,
        cancelled: &AtomicBool,
    ) -> Self {
        let mut text = String::with_capacity(columns.len());
        let mut fixed_spans = Vec::<(Range<u32>, Pixels)>::new();
        let mut segments = Vec::<StyleSegment>::new();
        let mut replaced = ReplacedChars::default();
        let mut replacement_widths = HashMap::<(char, FontOverrides), Pixels>::default();
        for chunk in shaper.chunks(snapshot, row, columns) {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }
            let start = text.len() as u32;
            text.push_str(chunk.text);
            let end = text.len() as u32;
            let overrides = FontOverrides::from_style(chunk.style);
            match segments.last_mut() {
                Some(segment) if segment.style == chunk.style => segment.end = end,
                last => {
                    let decorations = Decorations::from_run(&shaper.run(chunk.style, 0));
                    match last {
                        Some(segment)
                            if segment.overrides == overrides
                                && segment.decorations == decorations =>
                        {
                            segment.end = end
                        }
                        _ => segments.push(StyleSegment {
                            end,
                            overrides,
                            decorations,
                            style: chunk.style,
                        }),
                    }
                }
            }
            let Some(replacement) = chunk.replacement else {
                continue;
            };
            let width = match replacement {
                ChunkReplacement::Str(replacement) => {
                    let mut characters = chunk.text.chars();
                    if let (Some(character), None) = (characters.next(), characters.next()) {
                        replaced.insert(start);
                        replacement_widths
                            .entry((character, overrides))
                            .or_insert_with(|| shaper.uncached_width(&replacement, chunk.style));
                        continue;
                    }
                    shaper.uncached_width(&replacement, chunk.style)
                }
                ChunkReplacement::Renderer(renderer) => renderer
                    .measured_width
                    .or_else(|| match renderer.id {
                        ChunkRendererId::Inlay(id) => renderer_widths.get(&id).copied(),
                        ChunkRendererId::Fold(_) => None,
                    })
                    .unwrap_or_else(|| shaper.uncached_width(chunk.text, chunk.style)),
            };
            fixed_spans.push((start..end, width));
        }
        Self {
            text,
            fixed_spans,
            replaced,
            replacement_widths,
            segments,
            forced_boundaries: Vec::new(),
            hard_breaks: Vec::new(),
        }
    }

    fn is_replacement_edge(&self, offset: u32) -> bool {
        if self.replaced.is_empty() {
            return false;
        }
        self.replaced.contains(offset)
            || self.text[..offset as usize]
                .chars()
                .next_back()
                .is_some_and(|before| self.replaced.contains(offset - before.len_utf8() as u32))
    }

    fn max_chunk_end(&self, start: u32) -> u32 {
        start
            .saturating_add(MAX_CHUNK_LEN)
            .min(self.text.len() as u32)
    }

    fn assign_forced_boundaries(
        &mut self,
        from: u32,
        shaper: &RulerShaper,
        cancelled: &AtomicBool,
    ) {
        if self.max_chunk_end(from) < self.text.len() as u32 {
            (self.forced_boundaries, self.hard_breaks) =
                self.forced_boundaries(from, shaper, cancelled);
        }
    }

    fn forced_boundaries(
        &self,
        from: u32,
        shaper: &RulerShaper,
        cancelled: &AtomicBool,
    ) -> (Vec<u32>, Vec<u32>) {
        let boundaries = &GraphemeBoundaries::new(&self.text);
        let len = self.text.len() as u32;
        let mut forced = Vec::new();
        let mut hard_breaks = Vec::new();
        let mut previous = from;
        while previous < len && !cancelled.load(Ordering::Relaxed) {
            let window_end = self.max_chunk_end(previous);
            if window_end == len {
                break;
            }
            if let Some(safe) = self.last_shaping_boundary_in(previous, window_end, boundaries) {
                previous = safe;
                continue;
            }
            let slack_end = window_end.saturating_add(FORCED_BOUNDARY_SLACK).min(len);
            let (boundary, glyph_safe) = self
                .forced_boundary_in(previous, window_end, slack_end, boundaries, shaper)
                .unwrap_or_else(|| (self.ceil_char_boundary(slack_end), false));
            if boundary >= len {
                break;
            }
            forced.push(boundary);
            if !glyph_safe {
                hard_breaks.push(boundary);
            }
            previous = boundary;
        }
        (forced, hard_breaks)
    }

    fn forced_boundary_in(
        &self,
        previous: u32,
        start: u32,
        end: u32,
        boundaries: &GraphemeBoundaries,
        shaper: &RulerShaper,
    ) -> Option<(u32, bool)> {
        let context_start =
            self.ceil_char_boundary(start.saturating_sub(FORCED_BOUNDARY_CONTEXT).max(previous));
        let stretch_end = self
            .fixed_spans
            .get(
                self.fixed_spans
                    .partition_point(|(span, _)| span.start < start),
            )
            .map_or(self.text.len() as u32, |(span, _)| span.start);
        let layout_end = self.floor_char_boundary(
            end.saturating_add(FORCED_BOUNDARY_CONTEXT)
                .min(stretch_end)
                .max(end),
        );
        let layout = self.layout_range(context_start..layout_end, shaper);
        boundaries
            .next_matching(start, end, |offset| {
                offset < end && layout.is_glyph_boundary((offset - context_start) as usize)
            })
            .map(|offset| (offset, true))
            .or_else(|| {
                layout
                    .glyphs_in((start - context_start) as usize..(end - context_start) as usize)
                    .into_iter()
                    .map(|(_, index, _)| index as u32 + context_start)
                    .filter(|offset| self.text.is_char_boundary(*offset as usize))
                    .min()
                    .map(|offset| (offset, true))
            })
            .or_else(|| {
                boundaries
                    .next_matching(start, end, |_| true)
                    .map(|offset| (offset, false))
            })
    }

    fn last_shaping_boundary_in(
        &self,
        after: u32,
        end: u32,
        boundaries: &GraphemeBoundaries,
    ) -> Option<u32> {
        let bytes = self.text.as_bytes();
        let end = end.min(bytes.len() as u32);
        let fixed_edge = self.last_fixed_span_edge_in(after, end);
        let floor = fixed_edge.unwrap_or(after);
        let mut position = end;
        while position > floor {
            let offset = position as usize;
            let beside_isolated_byte = bytes[offset - 1..]
                .iter()
                .take(2)
                .any(|byte| may_be_shaping_isolated(*byte));
            if (beside_isolated_byte
                || self.text.is_char_boundary(offset) && self.is_replacement_edge(position))
                && self.is_shaping_boundary_between_graphemes(position)
                && boundaries.is_boundary(position)
            {
                return Some(position);
            }
            position -= 1;
        }
        fixed_edge
    }

    fn last_fixed_span_edge_in(&self, after: u32, end: u32) -> Option<u32> {
        let first_after_end = self
            .fixed_spans
            .partition_point(|(span, _)| span.start <= end);
        self.fixed_spans[..first_after_end]
            .iter()
            .rev()
            .take_while(|(span, _)| span.end > after)
            .flat_map(|(span, _)| [span.end, span.start])
            .find(|edge| after < *edge && *edge <= end)
    }

    fn is_forced_boundary(&self, offset: u32) -> bool {
        self.forced_boundaries.binary_search(&offset).is_ok()
    }

    fn context_reach(&self, offset: u32) -> u32 {
        if self.is_forced_boundary(offset) {
            FORCED_BOUNDARY_CONTEXT
        } else {
            SHAPING_CONTEXT
        }
    }

    fn add_forced_boundaries(&mut self, offsets: impl IntoIterator<Item = u32>) {
        self.forced_boundaries.extend(offsets);
        self.forced_boundaries.sort_unstable();
        self.forced_boundaries.dedup();
    }

    fn next_forced_boundary_after(&self, offset: u32) -> u32 {
        next_boundary_after(&self.forced_boundaries, offset, self.text.len() as u32)
    }

    fn is_shaping_boundary(&self, offset: u32, boundaries: &GraphemeBoundaries) -> bool {
        if offset == 0 || offset == self.text.len() as u32 || self.is_forced_boundary(offset) {
            return true;
        }
        boundaries.is_boundary(offset) && self.is_shaping_boundary_between_graphemes(offset)
    }

    fn is_shaping_boundary_between_graphemes(&self, offset: u32) -> bool {
        if !self.text.is_char_boundary(offset as usize) {
            return false;
        }
        if self.is_replacement_edge(offset) {
            return true;
        }
        let span_edge = self
            .fixed_spans
            .partition_point(|(span, _)| span.end < offset);
        if self.fixed_spans[span_edge..]
            .iter()
            .take_while(|(span, _)| span.start <= offset)
            .any(|(span, _)| span.start == offset || span.end == offset)
        {
            return true;
        }
        let offset = offset as usize;
        let (Some(before), Some(after)) = (
            self.text[..offset].chars().next_back(),
            self.text[offset..].chars().next(),
        ) else {
            return false;
        };
        is_shaping_isolated(before) || is_shaping_isolated(after)
    }

    fn context_bounds(&self, range: &Range<u32>) -> Range<u32> {
        let stretch_start = self
            .fixed_spans
            .partition_point(|(span, _)| span.end <= range.start);
        let stretch = stretch_start
            .checked_sub(1)
            .map_or(0, |ix| self.fixed_spans[ix].0.end)
            ..self
                .fixed_spans
                .get(stretch_start)
                .map_or(self.text.len() as u32, |(span, _)| span.start);
        stretch
            .start
            .max(last_boundary_at_or_before(&self.hard_breaks, range.start))
            ..stretch.end.min(next_boundary_after(
                &self.hard_breaks,
                range.end.saturating_sub(1),
                self.text.len() as u32,
            ))
    }

    fn context_of(&self, range: Range<u32>) -> Range<u32> {
        let bounds = self.context_bounds(&range);
        let start = range
            .start
            .saturating_sub(self.context_reach(range.start))
            .max(bounds.start);
        let end = range
            .end
            .saturating_add(self.context_reach(range.end))
            .min(bounds.end);
        self.ceil_char_boundary(start)..self.ceil_char_boundary(end)
    }

    fn ceil_char_boundary(&self, offset: u32) -> u32 {
        self.text.ceil_char_boundary(offset as usize) as u32
    }

    fn floor_char_boundary(&self, offset: u32) -> u32 {
        self.text.floor_char_boundary(offset as usize) as u32
    }

    fn segments_in(&self, range: Range<u32>) -> impl Iterator<Item = (Range<u32>, &StyleSegment)> {
        let first = self
            .segments
            .partition_point(|segment| segment.end <= range.start);
        let mut segments = self.segments[first..].iter();
        let mut start = first.checked_sub(1).map_or(0, |ix| self.segments[ix].end);
        std::iter::from_fn(move || {
            if start >= range.end {
                return None;
            }
            let segment = segments.next()?;
            let segment_range = start.max(range.start)..segment.end.min(range.end);
            start = segment.end;
            Some((segment_range, segment))
        })
    }

    fn hash_chunk(&self, range: Range<u32>, shaping: ChunkShaping) -> u64 {
        let mut hasher = FxHasher::default();
        let context = match shaping {
            ChunkShaping::Fixed(width) => {
                hasher.write_u8(1);
                hasher.write_u32(f32::from(width).to_bits());
                range
            }
            ChunkShaping::Context(context) => {
                hasher.write_u8(0);
                hasher.write_u32(range.start - context.start);
                hasher.write_u32(context.end - range.end);
                context
            }
        };
        hasher.write(&self.text.as_bytes()[context.start as usize..context.end as usize]);
        for replaced in self.replaced.in_range(context.clone()) {
            hasher.write_u32(replaced - context.start);
        }
        for (segment_range, segment) in self.segments_in(context) {
            hasher.write_u32(segment_range.end - segment_range.start);
            segment.overrides.hash(&mut hasher);
            segment.decorations.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn layout_range(&self, range: Range<u32>, shaper: &RulerShaper) -> ComposedLayout {
        let mut composed = ComposedLayout::default();
        let mut text_start = range.start;
        for start in self.replaced.in_range(range.clone()) {
            let Some(character) = self.text[start as usize..].chars().next() else {
                break;
            };
            if text_start < start {
                composed.push_text(self.layout_text(text_start..start, shaper));
            }
            let overrides = self
                .segments_in(start..start + 1)
                .next()
                .map(|(_, segment)| segment.overrides)
                .unwrap_or_default();
            let width = self
                .replacement_widths
                .get(&(character, overrides))
                .copied()
                .unwrap_or_default();
            composed.push_replacement(character.len_utf8(), width);
            text_start = start + character.len_utf8() as u32;
        }
        if text_start < range.end {
            composed.push_text(self.layout_text(text_start..range.end, shaper));
        }
        composed
    }

    fn layout_text(&self, range: Range<u32>, shaper: &RulerShaper) -> Arc<LineLayout> {
        let runs = self
            .segments_in(range.clone())
            .map(|(segment_range, segment)| {
                shaper.run(
                    segment.style,
                    (segment_range.end - segment_range.start) as usize,
                )
            })
            .collect::<Vec<_>>();
        shaper.layout_uncached(&self.text[range.start as usize..range.end as usize], &runs)
    }

    fn measure_chunk(
        &self,
        range: &Range<u32>,
        context: &Range<u32>,
        shaper: &RulerShaper,
    ) -> (Pixels, ComposedLayout) {
        let layout = self.layout_range(context.clone(), shaper);
        let start_x = chunk_origin(&layout, context, range.start);
        let end_x = if range.end < context.end {
            layout.x_for_index((range.end - context.start) as usize)
        } else {
            layout.width()
        };
        (end_x - start_x, layout)
    }
}

enum ChunkShaping {
    Fixed(Pixels),
    Context(Range<u32>),
}

struct ChunkSpan {
    range: Range<u32>,
    fixed_width: Option<Pixels>,
}

struct Neighbor {
    range: Range<u32>,
    context: Range<u32>,
    seam_after: bool,
}

impl Neighbor {
    fn of(chunk: &RulerChunk, start: u32) -> Option<Self> {
        (chunk.measured && !chunk.fixed).then(|| Self {
            range: start..start + chunk.len,
            context: start - chunk.context_before..start + chunk.len + chunk.context_after,
            seam_after: chunk.seam_after,
        })
    }

    fn shifted_back(self, offset: u32) -> Self {
        Self {
            range: self.range.start - offset..self.range.end - offset,
            context: self.context.start - offset..self.context.end - offset,
            ..self
        }
    }
}

struct MeasuredChunks {
    chunks: Vec<RulerChunk>,
    left: Option<RulerChunk>,
    right: Option<RulerChunk>,
    local: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Synchronization {
    Agreed,
    AgreedAfterExtending,
    Exhausted,
}

struct PendingChunk {
    range: Range<u32>,
    context: Range<u32>,
    fixed_width: Option<Pixels>,
    measured: Option<(Pixels, ComposedLayout)>,
    changed: bool,
    seam_after: bool,
}

impl PendingChunk {
    fn new(row_text: &RowText, span: ChunkSpan) -> Self {
        let context = match span.fixed_width {
            Some(_) => span.range.clone(),
            None => row_text.context_of(span.range.clone()),
        };
        Self {
            range: span.range,
            context,
            fixed_width: span.fixed_width,
            measured: None,
            changed: true,
            seam_after: false,
        }
    }

    fn neighbor(neighbor: Neighbor) -> Self {
        Self {
            range: neighbor.range,
            context: neighbor.context,
            fixed_width: None,
            measured: None,
            changed: false,
            seam_after: neighbor.seam_after,
        }
    }

    fn measured(&mut self, row_text: &RowText, shaper: &RulerShaper) -> &(Pixels, ComposedLayout) {
        self.measured
            .get_or_insert_with(|| row_text.measure_chunk(&self.range, &self.context, shaper))
    }

    fn set_context(&mut self, context: Range<u32>) {
        self.context = context;
        self.measured = None;
        self.changed = true;
    }

    fn into_changed_chunk(self, row_text: &RowText, shaper: &RulerShaper) -> Option<RulerChunk> {
        self.changed.then(|| self.into_chunk(row_text, shaper))
    }

    fn into_chunk(mut self, row_text: &RowText, shaper: &RulerShaper) -> RulerChunk {
        let len = self.range.end - self.range.start;
        if let Some(width) = self.fixed_width {
            return RulerChunk {
                len,
                width,
                hash: row_text.hash_chunk(self.range, ChunkShaping::Fixed(width)),
                fixed: true,
                context_before: 0,
                context_after: 0,
                measured: true,
                seam_after: false,
            };
        }
        let width = self.measured(row_text, shaper).0;
        RulerChunk {
            len,
            width,
            hash: row_text.hash_chunk(
                self.range.clone(),
                ChunkShaping::Context(self.context.clone()),
            ),
            fixed: false,
            context_before: self.range.start - self.context.start,
            context_after: self.context.end - self.range.end,
            measured: true,
            seam_after: self.seam_after,
        }
    }
}

fn next_boundary_after(boundaries: &[u32], offset: u32, len: u32) -> u32 {
    let ix = boundaries.partition_point(|boundary| *boundary <= offset);
    boundaries.get(ix).copied().unwrap_or(len)
}

fn last_boundary_at_or_before(boundaries: &[u32], offset: u32) -> u32 {
    let ix = boundaries.partition_point(|boundary| *boundary <= offset);
    ix.checked_sub(1).map_or(0, |ix| boundaries[ix])
}

fn may_be_shaping_isolated(byte: u8) -> bool {
    byte.is_ascii_whitespace() || byte >= 0x80
}

fn is_shaping_isolated(character: char) -> bool {
    character.is_whitespace()
        || matches!(
            u32::from(character),
            0x3000..=0x30FF
                | 0x3400..=0x4DBF
                | 0x4E00..=0x9FFF
                | 0xAC00..=0xD7AF
                | 0xF900..=0xFAFF
                | 0xFF00..=0xFFEF
                | 0x20000..=0x3134F
        )
}

impl RowRuler {
    pub(crate) fn new(
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        shaper: &RulerShaper,
        previous: Option<&RowRuler>,
        renderer_widths: &HashMap<InlayId, Pixels>,
    ) -> Self {
        Self::new_cancellable(
            snapshot,
            row,
            shaper,
            previous,
            renderer_widths,
            &AtomicBool::new(false),
        )
        .unwrap_or_else(|| {
            debug_panic!("an uncancelled ruler build always completes");
            Self::assemble(shaper.metrics_fingerprint(), None, Vec::new(), Vec::new())
        })
    }

    pub(crate) fn interpolate(&self, patches: &[Vec<RowEdit>]) -> Option<Self> {
        if self.has_seams() {
            return None;
        }
        let edits = compose_row_edits(patches.iter().map(Vec::as_slice))?;
        Some(Self::assemble(
            self.metrics_fingerprint,
            self.syntax.clone(),
            self.diagnostics.clone(),
            interpolate_chunks(&self.chunks, &edits)?,
        ))
    }

    pub(crate) fn needs_measuring(&self) -> bool {
        self.chunks.iter().any(|chunk| !chunk.measured)
    }

    pub(crate) fn unmeasured_len(&self) -> u32 {
        (0..self.chunks.len())
            .filter(|ix| !self.chunks[*ix].measured)
            .fold(0, |len, ix| {
                len.saturating_add(self.chunk_context(ix).len() as u32)
            })
    }

    pub(crate) fn measured(
        &self,
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        shaper: &RulerShaper,
        renderer_widths: &HashMap<InlayId, Pixels>,
    ) -> Option<Self> {
        self.measured_until(snapshot, row, shaper, renderer_widths, u32::MAX)
    }

    pub(crate) fn measured_until(
        &self,
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        shaper: &RulerShaper,
        renderer_widths: &HashMap<InlayId, Pixels>,
        until: u32,
    ) -> Option<Self> {
        self.measured_cancellable(
            snapshot,
            row,
            shaper,
            renderer_widths,
            until,
            &AtomicBool::new(false),
        )
    }

    pub(super) fn syntax(&self) -> Option<&SyntaxStamp> {
        self.syntax.as_ref()
    }

    pub(super) fn diagnostics(&self) -> &[DiagnosticKey] {
        &self.diagnostics
    }

    pub(crate) fn len(&self) -> u32 {
        self.starts.last().copied().unwrap_or(0)
    }

    pub(crate) fn width(&self) -> ScrollPixelOffset {
        self.xs.last().copied().unwrap_or(0.)
    }

    pub(crate) fn x_for_column(
        &self,
        column: u32,
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        shaper: &RulerShaper,
    ) -> ScrollPixelOffset {
        if column >= self.len() {
            return self.width();
        }
        let ix = self
            .starts
            .partition_point(|start| *start <= column)
            .saturating_sub(1);
        let start = self.starts[ix];
        let x = self.xs[ix];
        if self.chunks[ix].fixed || column == start {
            return x;
        }
        let context = self.chunk_context(ix);
        let layout = shaper.compose_columns(snapshot, row, context.clone());
        let origin = chunk_origin(&layout, &context, start);
        x + ScrollPixelOffset::from(layout.x_for_index((column - context.start) as usize) - origin)
    }

    pub(crate) fn column_for_x(
        &self,
        x: ScrollPixelOffset,
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        shaper: &RulerShaper,
    ) -> u32 {
        if self.chunks.is_empty() || x <= 0. {
            return 0;
        }
        if x >= self.width() {
            return self.len();
        }
        let ix = self
            .xs
            .partition_point(|chunk_x| *chunk_x <= x)
            .saturating_sub(1)
            .min(self.chunks.len() - 1);
        let start = self.starts[ix];
        let chunk_x = self.xs[ix];
        let chunk = &self.chunks[ix];
        if chunk.fixed {
            return if x - chunk_x > ScrollPixelOffset::from(chunk.width) / 2. {
                start + chunk.len
            } else {
                start
            };
        }
        let context = self.chunk_context(ix);
        let layout = shaper.compose_columns(snapshot, row, context.clone());
        let origin = chunk_origin(&layout, &context, start);
        let column =
            context.start + layout.closest_index_for_x(Pixels::from(x - chunk_x) + origin) as u32;
        column.clamp(start, start + chunk.len)
    }

    pub(crate) fn columns_for_x_range(&self, x: Range<ScrollPixelOffset>) -> Range<u32> {
        if self.chunks.is_empty() {
            return 0..0;
        }
        let last = self.chunks.len();
        let first = self
            .xs
            .partition_point(|chunk_x| *chunk_x <= x.start)
            .saturating_sub(1)
            .min(last - 1);
        let end = self
            .xs
            .partition_point(|chunk_x| *chunk_x < x.end)
            .clamp(first + 1, last);
        self.starts[first]..self.starts[end]
    }

    pub(crate) fn x_range_for_columns(&self, columns: Range<u32>) -> Range<ScrollPixelOffset> {
        let (first, end) = self.chunk_ixs_for_columns(columns);
        self.xs[first]..self.xs[end]
    }

    pub(crate) fn render_pieces(
        &self,
        columns: Range<u32>,
        visible_x: Range<ScrollPixelOffset>,
    ) -> Vec<RenderPiece> {
        let (first, end) = self.chunk_ixs_for_columns(columns);
        let mut pieces = (first..end)
            .map(|ix| RenderPiece {
                chunk: self.starts[ix]..self.starts[ix + 1],
                context: self.chunk_context(ix),
                width: self.chunks[ix].width,
                x: self.xs[ix],
                shaped: false,
            })
            .collect::<Vec<_>>();
        let visible = |piece: &RenderPiece| {
            piece.x < visible_x.end
                && visible_x.start <= piece.x + ScrollPixelOffset::from(piece.width)
        };
        let density = |piece: &RenderPiece| f32::from(piece.width) / piece.context.len() as f32;
        let mut by_priority = (0..pieces.len()).collect::<Vec<_>>();
        by_priority.sort_by(|a, b| {
            let (a, b) = (&pieces[*a], &pieces[*b]);
            visible(b)
                .cmp(&visible(a))
                .then_with(|| density(b).total_cmp(&density(a)))
        });
        let mut remaining = MAX_RENDERED_LEN;
        for ix in by_priority {
            let piece = &mut pieces[ix];
            if let Some(left) = remaining.checked_sub(piece.context.len() as u32) {
                remaining = left;
                piece.shaped = true;
            }
        }
        pieces
    }

    pub(crate) fn unmeasured(len: u32, shaper: &RulerShaper) -> Self {
        Self::assemble(
            shaper.metrics_fingerprint(),
            None,
            Vec::new(),
            vec![RulerChunk::unmeasured(len)],
        )
    }

    pub(crate) fn prefix(
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        shaper: &RulerShaper,
        previous: Option<&RowRuler>,
        renderer_widths: &HashMap<InlayId, Pixels>,
        limit: u32,
    ) -> Self {
        let row_len = snapshot.line_len(row);
        let limit = snapshot.clip_window_column(row, limit.min(row_len), row_len, Bias::Right);
        if limit >= row_len {
            return Self::new(snapshot, row, shaper, previous, renderer_widths);
        }
        let never_cancelled = AtomicBool::new(false);
        let row_text = RowText::read(
            snapshot,
            row,
            shaper,
            renderer_widths,
            0..limit,
            &never_cancelled,
        );
        let mut chunks =
            Self::chunk_row_text(row_text, previous, shaper, &never_cancelled).unwrap_or_default();
        let trusted_end = limit.saturating_sub(PREFIX_MARGIN);
        let mut trusted_len = 0;
        let trusted_count = chunks
            .iter()
            .take_while(|chunk| {
                let trusted =
                    chunk.measured && trusted_len + chunk.len + chunk.context_after <= trusted_end;
                if trusted {
                    trusted_len += chunk.len;
                }
                trusted
            })
            .count();
        chunks.truncate(trusted_count);
        chunks.push(RulerChunk::unmeasured(row_len - trusted_len));
        Self::stamped(snapshot, row, shaper, chunks)
    }

    pub(crate) fn measured_prefix(&self) -> (u32, ScrollPixelOffset) {
        let count = self
            .chunks
            .iter()
            .take_while(|chunk| chunk.measured)
            .count();
        (self.starts[count], self.xs[count])
    }

    fn new_cancellable(
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        shaper: &RulerShaper,
        previous: Option<&RowRuler>,
        renderer_widths: &HashMap<InlayId, Pixels>,
        cancelled: &AtomicBool,
    ) -> Option<Self> {
        let row_text = RowText::read(
            snapshot,
            row,
            shaper,
            renderer_widths,
            0..snapshot.line_len(row),
            cancelled,
        );
        let chunks = Self::chunk_row_text(row_text, previous, shaper, cancelled)?;
        Some(Self::stamped(snapshot, row, shaper, chunks))
    }

    fn chunk_row_text(
        mut row_text: RowText,
        previous: Option<&RowRuler>,
        shaper: &RulerShaper,
        cancelled: &AtomicBool,
    ) -> Option<Vec<RulerChunk>> {
        let is_cancelled = || cancelled.load(Ordering::Relaxed);
        if is_cancelled() {
            return None;
        }
        row_text.assign_forced_boundaries(0, shaper, cancelled);
        if is_cancelled() {
            return None;
        }
        let boundaries = GraphemeBoundaries::new(&row_text.text);
        let previous = previous.filter(|previous| !previous.has_seams());
        let (chunks, canonical) =
            Self::measure_row_text(&row_text, &boundaries, previous, shaper, cancelled)?;
        if canonical {
            Some(chunks)
        } else {
            Self::measure_row_text(&row_text, &boundaries, None, shaper, cancelled)
                .map(|(chunks, _)| chunks)
        }
    }

    fn measure_row_text(
        row_text: &RowText,
        boundaries: &GraphemeBoundaries,
        previous: Option<&RowRuler>,
        shaper: &RulerShaper,
        cancelled: &AtomicBool,
    ) -> Option<(Vec<RulerChunk>, bool)> {
        let (mut prefix, mut suffix) = previous.map_or((Vec::new(), Vec::new()), |previous| {
            previous.reusable_chunks(row_text, boundaries)
        });
        let reused = !prefix.is_empty() || !suffix.is_empty();
        let middle_start = prefix.iter().map(|chunk| chunk.len).sum::<u32>();
        let middle_end =
            row_text.text.len() as u32 - suffix.iter().map(|chunk| chunk.len).sum::<u32>();
        let left = prefix
            .last()
            .and_then(|chunk| Neighbor::of(chunk, middle_start - chunk.len));
        let right = suffix
            .first()
            .and_then(|chunk| Neighbor::of(chunk, middle_end));
        let spans = chunk_spans(row_text, middle_start..middle_end, boundaries, cancelled);
        let measured = measure_chunks(row_text, spans, left, right, shaper, cancelled);
        if cancelled.load(Ordering::Relaxed) {
            return None;
        }
        if let (Some(left), Some(last)) = (measured.left, prefix.last_mut()) {
            *last = left;
        }
        if let (Some(right), Some(first)) = (measured.right, suffix.first_mut()) {
            *first = right;
        }
        prefix.extend(measured.chunks);
        prefix.extend(suffix);
        Some((prefix, !reused || measured.local))
    }

    fn stamped(
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        shaper: &RulerShaper,
        chunks: Vec<RulerChunk>,
    ) -> Self {
        let buffer = snapshot.buffer_snapshot();
        let tab_row = snapshot.unwrapped_tab_row(row);
        let buffer_range = row_buffer_range(snapshot.fold_snapshot(), tab_row);
        Self::assemble(
            shaper.metrics_fingerprint(),
            shaper
                .language_aware
                .tree_sitter
                .then(|| SyntaxStamp::new(buffer, buffer_range.clone())),
            diagnostic_keys(buffer, buffer_range),
            chunks,
        )
    }

    fn measured_cancellable(
        &self,
        snapshot: &DisplaySnapshot,
        row: DisplayRow,
        shaper: &RulerShaper,
        renderer_widths: &HashMap<InlayId, Pixels>,
        until: u32,
        cancelled: &AtomicBool,
    ) -> Option<Self> {
        if self.len() != snapshot.line_len(row) || self.has_seams() {
            return None;
        }
        let mut chunks = Vec::<RulerChunk>::with_capacity(self.chunks.len());
        let mut measured_next = None;
        for (ix, chunk) in self.chunks.iter().enumerate() {
            if chunk.measured {
                chunks.push(measured_next.take().unwrap_or_else(|| chunk.clone()));
                continue;
            }
            if self.starts[ix + 1] > until {
                chunks.push(chunk.clone());
                continue;
            }
            let range = self.starts[ix]..self.starts[ix + 1];
            let left = chunks
                .last()
                .and_then(|last| Neighbor::of(last, range.start - last.len));
            let right = self
                .chunks
                .get(ix + 1)
                .and_then(|next| Neighbor::of(next, range.end));
            let row_len = self.len();
            let reach_start = range.start.saturating_sub(MAX_SHAPED_LEN);
            let aligned_start = self.starts[self
                .starts
                .partition_point(|start| *start <= reach_start)
                .saturating_sub(1)];
            let read_start = left
                .as_ref()
                .map_or(range.start, |left| left.context.start)
                .min(if reach_start - aligned_start > MAX_SHAPED_LEN {
                    snapshot.clip_window_column(row, reach_start, row_len, Bias::Left)
                } else {
                    aligned_start
                });
            let reach_end = range.end.saturating_add(MAX_SHAPED_LEN).min(row_len);
            let aligned_end = self.starts[self
                .starts
                .partition_point(|start| *start < reach_end)
                .min(self.starts.len() - 1)];
            let read_end = right
                .as_ref()
                .map_or(range.end, |right| right.context.end)
                .max(if aligned_end - reach_end > MAX_SHAPED_LEN {
                    snapshot.clip_window_column(row, reach_end, row_len, Bias::Right)
                } else {
                    aligned_end
                });
            let mut local = RowText::read(
                snapshot,
                row,
                shaper,
                renderer_widths,
                read_start..read_end,
                cancelled,
            );
            if local.text.len() != (read_end - read_start) as usize
                || contains_regional_indicators(&local.text)
            {
                return None;
            }
            let local_range = range.start - read_start..range.end - read_start;
            local.assign_forced_boundaries(local_range.start, shaper, cancelled);
            local.add_forced_boundaries([local_range.start, local_range.end]);
            let boundaries = GraphemeBoundaries::new(&local.text);
            let spans = chunk_spans(&local, local_range, &boundaries, cancelled);
            let measured = measure_chunks(
                &local,
                spans,
                left.map(|left| left.shifted_back(read_start)),
                right.map(|right| right.shifted_back(read_start)),
                shaper,
                cancelled,
            );
            if cancelled.load(Ordering::Relaxed) {
                return None;
            }
            if measured.chunks.iter().any(|chunk| {
                chunk.len + chunk.context_before + chunk.context_after > MAX_SHAPED_LEN
            }) {
                return None;
            }
            if !measured.local {
                return None;
            }
            if let (Some(left), Some(last)) = (measured.left, chunks.last_mut()) {
                *last = left;
            }
            chunks.extend(measured.chunks);
            measured_next = measured.right;
        }
        Some(Self::assemble(
            self.metrics_fingerprint,
            self.syntax.clone(),
            self.diagnostics.clone(),
            chunks,
        ))
    }

    fn has_seams(&self) -> bool {
        has_seams(&self.chunks)
    }

    fn chunk_ixs_for_columns(&self, columns: Range<u32>) -> (usize, usize) {
        let first = self
            .starts
            .partition_point(|start| *start <= columns.start)
            .saturating_sub(1)
            .min(self.chunks.len());
        let end = self
            .starts
            .partition_point(|start| *start < columns.end)
            .clamp(first, self.chunks.len());
        (first, end)
    }

    fn chunk_context(&self, ix: usize) -> Range<u32> {
        let chunk = &self.chunks[ix];
        self.starts[ix] - chunk.context_before..self.starts[ix + 1] + chunk.context_after
    }

    fn reusable_chunks(
        &self,
        row_text: &RowText,
        boundaries: &GraphemeBoundaries,
    ) -> (Vec<RulerChunk>, Vec<RulerChunk>) {
        let new_len = row_text.text.len() as u32;
        let mut prefix_count = 0;
        let mut position = 0;
        for chunk in &self.chunks {
            let end = position + chunk.len;
            if end > new_len
                || !chunk.measured
                || chunk_hash(row_text, position..end, boundaries) != Some(chunk.hash)
            {
                break;
            }
            position = end;
            prefix_count += 1;
        }
        if prefix_count == self.chunks.len() && position == new_len {
            return (self.chunks.clone(), Vec::new());
        }
        let mismatch_start = position;

        let shift = i64::from(new_len) - i64::from(self.len());
        let mut suffix_count = 0;
        for ix in (prefix_count..self.chunks.len()).rev() {
            let old_start = i64::from(self.starts[ix]) + shift;
            let old_end = i64::from(self.starts[ix + 1]) + shift;
            if old_start < i64::from(mismatch_start) {
                break;
            }
            let range = old_start as u32..old_end as u32;
            if !self.chunks[ix].measured
                || chunk_hash(row_text, range, boundaries) != Some(self.chunks[ix].hash)
            {
                break;
            }
            suffix_count += 1;
        }
        (
            self.chunks[..prefix_count].to_vec(),
            self.chunks[self.chunks.len() - suffix_count..].to_vec(),
        )
    }

    fn assemble(
        metrics_fingerprint: u64,
        syntax: Option<SyntaxStamp>,
        diagnostics: Vec<DiagnosticKey>,
        chunks: Vec<RulerChunk>,
    ) -> Self {
        let mut starts = Vec::with_capacity(chunks.len() + 1);
        let mut xs = Vec::with_capacity(chunks.len() + 1);
        starts.push(0);
        xs.push(0.);
        for chunk in &chunks {
            starts.push(starts.last().copied().unwrap_or(0) + chunk.len);
            xs.push(xs.last().copied().unwrap_or(0.) + ScrollPixelOffset::from(chunk.width));
        }
        Self {
            metrics_fingerprint,
            syntax,
            diagnostics,
            chunks,
            starts,
            xs,
        }
    }
}

fn interpolate_chunks(chunks: &[RulerChunk], edits: &[RowEdit]) -> Option<Vec<RulerChunk>> {
    let mut starts = Vec::with_capacity(chunks.len() + 1);
    let mut max_context_before = 0;
    let mut max_context_after = 0;
    starts.push(0u32);
    for chunk in chunks {
        starts.push(starts.last().copied().unwrap_or(0) + chunk.len);
        max_context_before = max_context_before.max(chunk.context_before);
        max_context_after = max_context_after.max(chunk.context_after);
    }
    let len = starts.last().copied().unwrap_or(0);
    let context = |ix: usize| {
        starts[ix].saturating_sub(chunks[ix].context_before)
            ..starts[ix + 1].saturating_add(chunks[ix].context_after)
    };
    let mut groups = Vec::<(Range<usize>, i64)>::new();
    for edit in edits {
        if edit.old.start > edit.old.end || edit.old.end > len {
            return None;
        }
        let touches = |ix: &usize| {
            let context = context(*ix);
            context.start <= edit.old.end && edit.old.start <= context.end
        };
        let lower = starts
            .partition_point(|start| start.saturating_add(max_context_after) < edit.old.start)
            .saturating_sub(1)
            .min(chunks.len());
        let mut touched = (lower..chunks.len())
            .take_while(|ix| starts[*ix] <= edit.old.end.saturating_add(max_context_before))
            .filter(touches);
        let first = touched.next()?;
        let end = touched.last().unwrap_or(first) + 1;
        match groups.last_mut() {
            Some((group, delta)) if first < group.end => {
                group.end = group.end.max(end);
                *delta += edit.delta();
            }
            _ => groups.push((first..end, edit.delta())),
        }
    }
    let mut interpolated = Vec::with_capacity(chunks.len());
    let mut next = 0;
    for (group, delta) in groups {
        let merged = chunks.get(group.clone())?;
        let (Some(first), Some(last)) = (merged.first(), merged.last()) else {
            return None;
        };
        if group.start < next {
            return None;
        }
        let merged_len = i64::from(starts[group.end] - starts[group.start]) + delta;
        let merged_len = u32::try_from(merged_len).ok().filter(|len| *len > 0)?;
        interpolated.extend_from_slice(&chunks[next..group.start]);
        interpolated.push(RulerChunk {
            context_before: first.context_before,
            context_after: last.context_after,
            ..RulerChunk::unmeasured(merged_len)
        });
        next = group.end;
    }
    interpolated.extend_from_slice(chunks.get(next..)?);
    Some(interpolated)
}

fn has_seams(chunks: &[RulerChunk]) -> bool {
    chunks.iter().any(|chunk| chunk.seam_after)
}

fn contains_regional_indicators(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.contains(&REGIONAL_INDICATOR_PREFIX[0])
        && bytes
            .windows(REGIONAL_INDICATOR_PREFIX.len())
            .any(|window| window == REGIONAL_INDICATOR_PREFIX)
}

fn chunk_origin(layout: &ComposedLayout, context: &Range<u32>, start: u32) -> Pixels {
    if context.start < start {
        layout.x_for_index((start - context.start) as usize)
    } else {
        Pixels::ZERO
    }
}

fn measure_chunks(
    row_text: &RowText,
    spans: Vec<ChunkSpan>,
    left: Option<Neighbor>,
    right: Option<Neighbor>,
    shaper: &RulerShaper,
    cancelled: &AtomicBool,
) -> MeasuredChunks {
    let has_left = left.is_some();
    let has_right = right.is_some();
    let mut pending = Vec::with_capacity(spans.len() + 2);
    pending.extend(left.map(PendingChunk::neighbor));
    pending.extend(
        spans
            .into_iter()
            .map(|span| PendingChunk::new(row_text, span)),
    );
    pending.extend(right.map(PendingChunk::neighbor));
    let mut chunks = Vec::with_capacity(pending.len());
    let mut pending = pending.into_iter();
    let mut previous = pending.next();
    let mut local = true;
    for mut next in pending {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        if let Some(left) = previous.as_mut() {
            let synchronization = synchronize(row_text, left, &mut next, shaper);
            local &= synchronization == Synchronization::Agreed;
            let seam = synchronization == Synchronization::Exhausted;
            if left.seam_after != seam {
                left.seam_after = seam;
                left.changed = true;
            }
        }
        if let Some(done) = previous.replace(next) {
            chunks.push(done.into_changed_chunk(row_text, shaper));
        }
    }
    chunks.extend(previous.map(|chunk| chunk.into_changed_chunk(row_text, shaper)));
    let right = if has_right {
        chunks.pop().flatten()
    } else {
        None
    };
    let left = if has_left && !chunks.is_empty() {
        chunks.remove(0)
    } else {
        None
    };
    MeasuredChunks {
        chunks: chunks.into_iter().flatten().collect(),
        left,
        right,
        local,
    }
}

fn synchronize(
    row_text: &RowText,
    left: &mut PendingChunk,
    right: &mut PendingChunk,
    shaper: &RulerShaper,
) -> Synchronization {
    if left.fixed_width.is_some() || right.fixed_width.is_some() {
        return Synchronization::Agreed;
    }
    let boundary = left.range.end;
    let bounds = row_text.context_bounds(&(left.range.start..right.range.end));
    let mut agreed = Synchronization::Agreed;
    loop {
        left.measured(row_text, shaper);
        right.measured(row_text, shaper);
        if let (Some((_, left_layout)), Some((_, right_layout))) = (&left.measured, &right.measured)
            && boundaries_agree(
                &left.context,
                left_layout,
                &right.context,
                right_layout,
                boundary,
            )
        {
            return agreed;
        }
        let reach = (left.context.end - boundary)
            .max(boundary - right.context.start)
            .max(SHAPING_CONTEXT)
            .saturating_mul(SYNC_GROWTH);
        let left_end = row_text.floor_char_boundary(
            boundary
                .saturating_add(reach)
                .min(bounds.end)
                .min(left.context.start + MAX_SHAPED_LEN)
                .max(left.context.end),
        );
        let right_start = row_text.ceil_char_boundary(
            boundary
                .saturating_sub(reach)
                .max(bounds.start)
                .max(right.context.end.saturating_sub(MAX_SHAPED_LEN))
                .min(right.context.start),
        );
        if left_end == left.context.end && right_start == right.context.start {
            return Synchronization::Exhausted;
        }
        agreed = Synchronization::AgreedAfterExtending;
        if left_end != left.context.end {
            left.set_context(left.context.start..left_end);
        }
        if right_start != right.context.start {
            right.set_context(right_start..right.context.end);
        }
    }
}

fn boundaries_agree(
    left_context: &Range<u32>,
    left: &ComposedLayout,
    right_context: &Range<u32>,
    right: &ComposedLayout,
    boundary: u32,
) -> bool {
    let compared = right_context
        .start
        .saturating_add(SYNC_MARGIN)
        .max(left_context.start)
        ..left_context
            .end
            .saturating_sub(SYNC_MARGIN)
            .min(right_context.end);
    if compared.start > boundary || boundary >= compared.end {
        return true;
    }
    let local = |context: &Range<u32>, offset: u32| (offset - context.start) as usize;
    if !left.is_glyph_boundary(local(left_context, boundary))
        || !right.is_glyph_boundary(local(right_context, boundary))
    {
        return false;
    }
    let glyphs = |context: &Range<u32>, layout: &ComposedLayout| {
        let start = context.start as usize;
        let mut previous_x = None;
        layout
            .glyphs_in(local(context, compared.start)..local(context, compared.end))
            .into_iter()
            .map(move |(glyph, index, x)| {
                let advance = previous_x.replace(x).map(|previous_x| x - previous_x);
                (glyph, index + start, advance)
            })
    };
    let mut left_glyphs = glyphs(left_context, left);
    let mut right_glyphs = glyphs(right_context, right);
    loop {
        match (left_glyphs.next(), right_glyphs.next()) {
            (None, None) => return true,
            (
                Some((left_glyph, left_index, left_advance)),
                Some((right_glyph, right_index, right_advance)),
            ) if left_glyph == right_glyph
                && left_index == right_index
                && advances_agree(left_advance, right_advance) => {}
            _ => return false,
        }
    }
}

fn advances_agree(left: Option<Pixels>, right: Option<Pixels>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => (f32::from(left) - f32::from(right)).abs() <= SYNC_TOLERANCE,
        (None, None) => true,
        _ => false,
    }
}

fn chunk_spans(
    row_text: &RowText,
    range: Range<u32>,
    boundaries: &GraphemeBoundaries,
    cancelled: &AtomicBool,
) -> Vec<ChunkSpan> {
    let mut chunks = Vec::new();
    let mut position = range.start;
    let mut spans = row_text
        .fixed_spans
        .iter()
        .skip_while(|(span, _)| span.end <= range.start)
        .peekable();
    while position < range.end && !cancelled.load(Ordering::Relaxed) {
        if let Some((span, width)) = spans.peek()
            && span.start <= position
        {
            let end = span.end.min(range.end);
            chunks.push(ChunkSpan {
                range: position..end,
                fixed_width: Some(*width),
            });
            position = end;
            spans.next();
            continue;
        }
        let stretch_end = spans
            .peek()
            .map_or(range.end, |(span, _)| span.start.min(range.end));
        let limit = stretch_end.min(row_text.next_forced_boundary_after(position));
        let target = position.saturating_add(CHUNK_LEN).min(limit);
        let end = if target >= limit {
            limit
        } else {
            boundaries
                .next_matching(target, limit, |offset| {
                    offset == limit || row_text.is_shaping_boundary_between_graphemes(offset)
                })
                .unwrap_or(limit)
        };
        chunks.push(ChunkSpan {
            range: position..end,
            fixed_width: None,
        });
        position = end;
    }
    chunks
}

fn chunk_hash(
    row_text: &RowText,
    range: Range<u32>,
    boundaries: &GraphemeBoundaries,
) -> Option<u64> {
    if !row_text.is_shaping_boundary(range.start, boundaries)
        || !row_text.is_shaping_boundary(range.end, boundaries)
    {
        return None;
    }
    let fixed_spans = &row_text.fixed_spans;
    let first_overlapping = fixed_spans.partition_point(|(span, _)| span.end <= range.start);
    let overlapping = fixed_spans[first_overlapping..]
        .iter()
        .take_while(|(span, _)| span.start < range.end)
        .collect::<Vec<_>>();
    let shaping = match overlapping.as_slice() {
        [] => ChunkShaping::Context(row_text.context_of(range.clone())),
        [(span, width)] if *span == range => ChunkShaping::Fixed(*width),
        _ => return None,
    };
    Some(row_text.hash_chunk(range, shaping))
}

const REGIONAL_INDICATOR_PREFIX: &[u8] = b"\xF0\x9F\x87";

enum GraphemeBoundaries<'a> {
    Ascii { len: u32 },
    Segmented { text: &'a str },
    Indexed { len: u32, bits: Vec<u64> },
}

impl<'a> GraphemeBoundaries<'a> {
    fn new(text: &'a str) -> Self {
        let len = text.len() as u32;
        if text.is_ascii() {
            return Self::Ascii { len };
        }
        if !contains_regional_indicators(text) {
            return Self::Segmented { text };
        }
        let mut bits = vec![0u64; text.len() / 64 + 1];
        for (offset, _) in text.grapheme_indices(true) {
            bits[offset / 64] |= 1 << (offset % 64);
        }
        bits[text.len() / 64] |= 1 << (text.len() % 64);
        Self::Indexed { len, bits }
    }

    fn len(&self) -> u32 {
        match self {
            Self::Ascii { len } | Self::Indexed { len, .. } => *len,
            Self::Segmented { text } => text.len() as u32,
        }
    }

    fn is_boundary(&self, offset: u32) -> bool {
        if offset > self.len() {
            return false;
        }
        match self {
            Self::Ascii { .. } => true,
            Self::Segmented { text } => {
                let offset = offset as usize;
                text.is_char_boundary(offset)
                    && GraphemeCursor::new(offset, text.len(), true).is_boundary(text, 0)
                        == Ok(true)
            }
            Self::Indexed { bits, .. } => {
                let offset = offset as usize;
                bits[offset / 64] & (1 << (offset % 64)) != 0
            }
        }
    }

    fn next_matching(
        &self,
        offset: u32,
        limit: u32,
        mut accept: impl FnMut(u32) -> bool,
    ) -> Option<u32> {
        let limit = limit.min(self.len());
        let offset = offset.min(limit);
        match self {
            Self::Ascii { .. } => (offset..=limit).find(|offset| accept(*offset)),
            Self::Segmented { text } => {
                let boundary = text.ceil_char_boundary(offset as usize);
                let limit = text.floor_char_boundary(limit as usize);
                if boundary > limit {
                    return None;
                }
                let searchable = &text[..limit];
                let mut cursor = GraphemeCursor::new(boundary, text.len(), true);
                if boundary == limit || cursor.is_boundary(searchable, 0) == Ok(true) {
                    if accept(boundary as u32) {
                        return Some(boundary as u32);
                    }
                    if boundary == limit {
                        return None;
                    }
                }
                loop {
                    match cursor.next_boundary(searchable, 0) {
                        Ok(Some(boundary)) if accept(boundary as u32) => {
                            return Some(boundary as u32);
                        }
                        Ok(Some(_)) => {}
                        Ok(None) => return accept(limit as u32).then_some(limit as u32),
                        Err(_) => return None,
                    }
                }
            }
            Self::Indexed { bits, .. } => {
                let mut offset = offset as usize;
                let limit = limit as usize;
                loop {
                    let mut word_ix = offset / 64;
                    let mut word = bits[word_ix] & (u64::MAX << (offset % 64));
                    while word == 0 && (word_ix + 1) * 64 <= limit {
                        word_ix += 1;
                        word = bits[word_ix];
                    }
                    if word == 0 {
                        return None;
                    }
                    let boundary = word_ix * 64 + word.trailing_zeros() as usize;
                    if boundary > limit {
                        return None;
                    }
                    if accept(boundary as u32) {
                        return Some(boundary as u32);
                    }
                    offset = boundary + 1;
                    if offset > limit {
                        return None;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Anchor, Inlay, SelectionGoal,
        display_map::{
            EstimatedRow, GridCell, HighlightKey, HorizontalViewport,
            tests::{displayed_row_ruler, init_test, long_row_test_map, test_shaper},
        },
        hover_links::InlayHighlight,
        test::editor_test_context::EditorTestContext,
    };
    use gpui::{
        Bounds, DevicePixels, FontFeatures, FontMetrics, FontRun, NoopTextSystem,
        PlatformTextSystem, RenderGlyphParams, ShapedGlyph, ShapedRun, Size, TestAppContext,
        TextAlign, TextRenderingMode, TextSystem, point, px,
    };
    use language::Point;
    use multi_buffer::MultiBufferOffset;
    use rand::prelude::*;
    use std::{borrow::Cow, iter};

    #[gpui::test]
    async fn test_ruler_chunks_match_full_row_layout(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let text = format!(
            "{}\u{1}{}e\u{301}{}",
            "e\u{301}".repeat(CHUNK_LEN as usize / 3 + 5),
            "🙂".repeat(MAX_LINE_LEN),
            "x".repeat(3_000)
        );
        let (ruler, snapshot, shaper) = ruler_for_text(&mut cx, &text);
        let full_row = shaper.compose_columns(&snapshot, DisplayRow(0), 0..text.len() as u32);

        assert_eq!(ruler.len(), text.len() as u32);
        assert!(ruler.chunks.len() >= 3);
        assert_eq!(ruler.chunks.iter().filter(|chunk| chunk.fixed).count(), 0);
        let boundaries = GraphemeBoundaries::new(&text);
        assert!(
            ruler
                .starts
                .iter()
                .all(|start| boundaries.is_boundary(*start))
        );
        let tolerance = ScrollPixelOffset::from(full_row.width()) * 1e-4;
        assert!((ruler.width() - ScrollPixelOffset::from(full_row.width())).abs() < tolerance);
        for column in (0..=text.len())
            .step_by(97)
            .filter(|column| text.is_char_boundary(*column))
        {
            let x = ruler.x_for_column(column as u32, &snapshot, DisplayRow(0), &shaper);
            let full_x = ScrollPixelOffset::from(full_row.x_for_index(column));
            assert!(
                (x - full_x).abs() < tolerance,
                "column {column}: {x} != {full_x}"
            );
            assert_eq!(
                ruler.column_for_x(x, &snapshot, DisplayRow(0), &shaper),
                column as u32
            );
        }
    }

    #[gpui::test]
    async fn test_ruler_reuses_unchanged_chunks_across_edits(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let text = "漢字".repeat(MAX_LINE_LEN * 4);
        let (before, _, shaper) = ruler_for_text(&mut cx, &text);
        insert(&mut cx, Point::new(0, text.len() as u32 / 2), "ab");
        let snapshot =
            cx.update_editor(|editor, window, cx| editor.snapshot(window, cx).display_snapshot);
        let reused = RowRuler::new(
            &snapshot,
            DisplayRow(0),
            &shaper,
            Some(&before),
            &HashMap::default(),
        );
        let fresh = fresh_ruler(&snapshot, DisplayRow(0), &shaper);
        let edited_text = snapshot.text();
        let boundaries = GraphemeBoundaries::new(&edited_text);
        assert_eq!(reused.len(), edited_text.len() as u32);
        assert!(
            reused
                .starts
                .iter()
                .all(|start| boundaries.is_boundary(*start))
        );
        assert!((reused.width() - fresh.width()).abs() < fresh.width() * 1e-6);
        assert!(before.chunks.len() >= 8);
        let same_hash = |(old, new): &(&RulerChunk, &RulerChunk)| old.hash == new.hash;
        let prefix = before
            .chunks
            .iter()
            .zip(&reused.chunks)
            .take_while(same_hash)
            .count();
        let suffix = before
            .chunks
            .iter()
            .rev()
            .zip(reused.chunks.iter().rev())
            .take_while(same_hash)
            .count();
        assert!(prefix > 0 && suffix > 0);
        assert!(reused.chunks.len() - prefix - suffix <= 4);
    }

    #[gpui::test]
    async fn test_rulers_follow_edits_and_highlights(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let long = "漢字".repeat(MAX_LINE_LEN);
        cx.set_state(&format!("short\nˇ{long}\nother"));
        let highlight = |cx: &mut EditorTestContext, key, row: u32, columns: Range<u32>, style| {
            cx.update_editor(|editor, _, cx| {
                let buffer = editor.buffer().read(cx).snapshot(cx);
                let range = buffer.anchor_before(Point::new(row, columns.start))
                    ..buffer.anchor_after(Point::new(row, columns.end));
                editor.highlight_text_key(key, vec![range], style, false, cx);
            });
        };
        let clear = |cx: &mut EditorTestContext, key| {
            cx.update_editor(|editor, _, cx| editor.clear_highlights(key, cx));
        };
        let background = HighlightStyle {
            background_color: Some(gpui::red()),
            ..HighlightStyle::default()
        };
        let (read, write, search) = (
            HighlightKey::DocumentHighlightRead,
            HighlightKey::DocumentHighlightWrite,
            HighlightKey::BufferSearchHighlights,
        );
        let bold_columns = 3_000..3_012;

        let first = displayed_row_ruler(&mut cx, 1);
        insert(&mut cx, Point::new(2, 0), "x");
        assert!(Arc::ptr_eq(&first, &displayed_row_ruler(&mut cx, 1)));
        insert(&mut cx, Point::new(0, 0), "above\n");
        assert!(Arc::ptr_eq(&first, &displayed_row_ruler(&mut cx, 2)));
        highlight(&mut cx, read, 3, 0..1, bold());
        assert!(Arc::ptr_eq(&first, &displayed_row_ruler(&mut cx, 2)));
        highlight(&mut cx, write, 2, bold_columns.clone(), background);
        assert!(Arc::ptr_eq(&first, &displayed_row_ruler(&mut cx, 2)));
        clear(&mut cx, write);
        assert!(Arc::ptr_eq(&first, &displayed_row_ruler(&mut cx, 2)));

        highlight(&mut cx, search, 2, bold_columns.clone(), bold());
        let bolded = displayed_row_ruler(&mut cx, 2);
        let changed = (0..first.chunks.len())
            .filter(|ix| first.chunks[*ix].hash != bolded.chunks[*ix].hash)
            .collect::<Vec<_>>();
        let affected = (0..first.chunks.len())
            .filter(|ix| {
                let context = first.chunk_context(*ix);
                context.start < bold_columns.end && bold_columns.start < context.end
            })
            .collect::<Vec<_>>();
        assert_eq!((&first.starts, &changed), (&bolded.starts, &affected));
        assert!(!affected.is_empty() && affected.len() <= 2);
        clear(&mut cx, search);
        let unbolded = displayed_row_ruler(&mut cx, 2);
        assert!(!Arc::ptr_eq(&bolded, &unbolded));

        insert(&mut cx, Point::new(2, 3), "漢");
        let edited = displayed_row_ruler(&mut cx, 2);
        assert!(!Arc::ptr_eq(&unbolded, &edited));
        assert_eq!(edited.len(), long.len() as u32 + 3);
        insert(&mut cx, Point::new(2, 0), "\n");
        assert!(Arc::ptr_eq(&edited, &displayed_row_ruler(&mut cx, 3)));
        let row_end = Point::new(3, long.len() as u32 + 3);
        insert(&mut cx, row_end, "\n");
        assert!(Arc::ptr_eq(&edited, &displayed_row_ruler(&mut cx, 3)));
        insert(&mut cx, row_end, "x\n");
        assert!(!Arc::ptr_eq(&edited, &displayed_row_ruler(&mut cx, 3)));
    }

    #[gpui::test]
    async fn test_rows_chunk_only_beside_isolated_chars_or_at_hard_breaks(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let spaced = format!(
            "{} {},{}",
            "x".repeat(5_000),
            "y".repeat(3_000),
            "z".repeat(3_000)
        );
        let ligatures = format!("\u{754c}aa{}", "!=".repeat(1_050));
        let punctuation = "!=".repeat(10_000);
        for (text, inner_starts, inner_context) in [
            (&spaced, vec![5_000], SHAPING_CONTEXT),
            (&ligatures, Vec::new(), SHAPING_CONTEXT),
            (
                &punctuation,
                vec![MAX_CHUNK_LEN, 2 * MAX_CHUNK_LEN, 3 * MAX_CHUNK_LEN],
                FORCED_BOUNDARY_CONTEXT,
            ),
        ] {
            let (ruler, ..) = ruler_for_text(&mut cx, text);
            let mut starts = vec![0];
            starts.extend(inner_starts);
            starts.push(text.len() as u32);
            assert_eq!(ruler.starts, starts);
            let contexts = ruler
                .chunks
                .iter()
                .map(|chunk| (chunk.context_before, chunk.context_after))
                .collect::<Vec<_>>();
            let last = ruler.chunks.len() - 1;
            let expected = (0..=last)
                .map(|ix| {
                    (
                        if ix == 0 { 0 } else { inner_context },
                        if ix == last { 0 } else { inner_context },
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(contexts, expected);
        }
    }

    #[gpui::test]
    async fn test_forced_boundaries_keep_shaping_context(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let fraction = format!("{}1/{}", "2".repeat(6_000), "2".repeat(3_000));
        for text in ["AV".repeat(4_096), fraction] {
            let (snapshot, shaper) = snapshot_for_text(&mut cx, &text);
            let shaper = contextual_shaper(shaper);
            let ruler = fresh_ruler(&snapshot, DisplayRow(0), &shaper);
            let full_row = shaper.compose_columns(&snapshot, DisplayRow(0), 0..text.len() as u32);

            assert_eq!(ruler.starts, [0, MAX_CHUNK_LEN, text.len() as u32]);
            let full_width = ScrollPixelOffset::from(full_row.width());
            assert!(
                (ruler.width() - full_width).abs() < 0.1,
                "{} != {full_width}",
                ruler.width()
            );
        }

        let widths = |cx: &mut EditorTestContext| {
            cx.update_editor(|editor, window, cx| {
                let snapshot = editor.snapshot(window, cx).display_snapshot;
                let row = DisplayRow(0);
                let shaper = contextual_shaper(
                    editor
                        .text_layout_details(window, cx)
                        .ruler_shaper(&snapshot, row),
                );
                let cached = snapshot.ruled_row(row, shaper.clone()).unwrap().ruler;
                let full_row = shaper.compose_columns(&snapshot, row, 0..snapshot.line_len(row));
                (cached.width(), ScrollPixelOffset::from(full_row.width()))
            })
        };
        widths(&mut cx);
        let row_end = Point::new(0, 9_002);
        let slash = Point::new(0, 6_001)..Point::new(0, 6_002);
        for edit in [(row_end..row_end, "2"), (slash, "x")] {
            cx.update_editor(|editor, _, cx| editor.edit([edit], cx));
            let (cached_width, full_width) = widths(&mut cx);
            assert!(
                (cached_width - full_width).abs() < 0.1,
                "{cached_width} != {full_width}"
            );
        }
    }

    #[gpui::test]
    async fn test_forced_boundaries_reach_distant_shaping_context(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let text = format!("1/{}", "2".repeat(8_190));
        let (snapshot, shaper) = snapshot_for_text(&mut cx, &text);
        let shaper = contextual_shaper(shaper);
        let ruler = fresh_ruler(&snapshot, DisplayRow(0), &shaper);
        let em = f64::from(f32::from(shaper.font_size)) * 0.6;
        let expected_x = |column: u32| match column {
            0..=2 => em * f64::from(column),
            _ => em * 2. + em / 2. * f64::from(column - 2),
        };

        assert!(ruler.starts.len() > 2);
        for column in [
            100,
            MAX_CHUNK_LEN,
            MAX_CHUNK_LEN + 1,
            8_000,
            text.len() as u32,
        ] {
            let x = ruler.x_for_column(column, &snapshot, DisplayRow(0), &shaper);
            assert!(
                (x - expected_x(column)).abs() < 0.1,
                "column {column}: {x} != {}",
                expected_x(column)
            );
        }
    }

    #[gpui::test]
    async fn test_ruler_edits_invalidate_expanded_shaping_context(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let text = format!("1/{}", "2".repeat(8_190));
        for (old, expected_width) in [(1..2, Some(49_152.)), (8_191..8_192, None)] {
            let (snapshot, shaper) = snapshot_for_text(&mut cx, &text);
            let mut shaper = contextual_shaper(shaper);
            shaper.font_size = px(10.);
            let ruler = fresh_ruler(&snapshot, DisplayRow(0), &shaper);
            assert_eq!(ruler.width(), 24_582.);
            let interpolated = ruler
                .interpolate(&[vec![RowEdit {
                    old: old.clone(),
                    new_len: 1,
                }]])
                .expect("the replacement preserves the row");
            assert_eq!(
                interpolated
                    .chunks
                    .iter()
                    .map(|chunk| (chunk.len, chunk.measured))
                    .collect::<Vec<_>>(),
                [(8_192, false)]
            );
            cx.update_editor(|editor, _, cx| {
                editor.edit(
                    [(Point::new(0, old.start)..Point::new(0, old.end), "x")],
                    cx,
                );
            });
            let snapshot =
                cx.update_editor(|editor, window, cx| editor.snapshot(window, cx).display_snapshot);
            let measured = interpolated
                .measured(&snapshot, DisplayRow(0), &shaper, &HashMap::default())
                .map(|measured| measured.width());
            assert_eq!(measured, expected_width);
        }
    }

    #[gpui::test]
    async fn test_rulers_with_out_of_reach_shaping_context_rebuild_like_fresh_rulers(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        snapshot_for_text(&mut cx, &format!("1/{}", "2".repeat(16_000)));
        let cached_and_fresh = |cx: &mut EditorTestContext| {
            cx.update_editor(|editor, window, cx| {
                let snapshot = editor.snapshot(window, cx).display_snapshot;
                let row = DisplayRow(0);
                let shaper = contextual_shaper(
                    editor
                        .text_layout_details(window, cx)
                        .ruler_shaper(&snapshot, row),
                );
                let cached = snapshot.ruled_row(row, shaper.clone()).unwrap().ruler;
                (cached, fresh_ruler(&snapshot, row, &shaper))
            })
        };

        let (cached, fresh) = cached_and_fresh(&mut cx);
        assert!(fresh.has_seams());
        assert_eq!((&cached.starts, &cached.xs), (&fresh.starts, &fresh.xs));
        insert(&mut cx, Point::new(0, 9_000), " ");
        let (cached, fresh) = cached_and_fresh(&mut cx);
        assert!(fresh.has_seams());
        assert_eq!((&cached.starts, &cached.xs), (&fresh.starts, &fresh.xs));
    }

    #[gpui::test]
    async fn test_grid_cells_reject_fonts_with_contextual_ascii_advances(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let (_, map) = long_row_test_map(&"x".repeat(2_000), cx);
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let shaper = contextual_shaper(test_shaper(cx));
        let text_system = &shaper.text_system;
        let font_id = text_system.resolve_font(&shaper.style.text.font());
        let mut advances = (0x20u8..=0x7E)
            .map(|byte| text_system.layout_width(font_id, shaper.font_size, char::from(byte)))
            .collect::<Vec<_>>();
        advances.dedup();
        assert_eq!(advances.len(), 1);
        assert!(
            !GridCell::measure(text_system, &shaper.style, &snapshot, shaper.font_size).monospace
        );
    }

    #[gpui::test]
    async fn test_vertical_moves_into_unmeasured_rows_stay_put_after_measurement(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let wide = "\u{1F600}".repeat(5_000);
        assert!(wide.len() as u32 > MAX_SYNC_RULER_LEN);
        cx.set_state(&format!("ˇ{}\n{wide}", "x".repeat(3_100)));
        let move_down = |cx: &mut EditorTestContext| {
            cx.update_editor(|editor, window, cx| {
                let snapshot = editor.snapshot(window, cx).display_snapshot;
                let details = editor.text_layout_details(window, cx);
                let start = DisplayPoint::new(DisplayRow(0), 3_050);
                crate::movement::down(&snapshot, start, SelectionGoal::None, false, &details).0
            })
        };
        let is_measured = |cx: &mut EditorTestContext| {
            cx.update_editor(|editor, window, cx| {
                let snapshot = editor.snapshot(window, cx).display_snapshot;
                let shaper = editor
                    .text_layout_details(window, cx)
                    .ruler_shaper(&snapshot, DisplayRow(1));
                snapshot
                    .ruled_row(DisplayRow(1), shaper)
                    .is_some_and(|ruled| ruled.exact)
            })
        };

        assert!(!is_measured(&mut cx));
        assert_eq!(
            move_down(&mut cx),
            DisplayPoint::new(DisplayRow(1), 4 * 3_050 / 2)
        );
        insert(&mut cx, Point::new(1, 0), "xx");
        let expected = DisplayPoint::new(DisplayRow(1), 2 + 4 * 3_048 / 2);
        assert!(!is_measured(&mut cx));
        assert_eq!(move_down(&mut cx), expected);
        build_queued_rulers(&mut cx);
        assert!(is_measured(&mut cx));
        assert_eq!(move_down(&mut cx), expected);
    }

    #[gpui::test]
    async fn test_rulers_follow_decoration_only_shaping_changes(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let text = "fi".repeat(MAX_LINE_LEN);
        cx.set_state(&format!("ˇ{text}"));
        let highlight = |cx: &mut EditorTestContext, key, parity: u32, style| {
            cx.update_editor(|editor, _, cx| {
                let buffer = editor.buffer().read(cx).snapshot(cx);
                let ranges = (0..MAX_LINE_LEN as u32)
                    .map(|pair| {
                        let column = 2 * pair + parity;
                        buffer.anchor_before(Point::new(0, column))
                            ..buffer.anchor_after(Point::new(0, column + 1))
                    })
                    .collect();
                editor.highlight_text_key(key, ranges, style, false, cx);
            });
        };
        let rulers = |cx: &mut EditorTestContext| {
            cx.update_editor(|editor, window, cx| {
                let snapshot = editor.snapshot(window, cx).display_snapshot;
                let shaper = contextual_shaper(
                    editor
                        .text_layout_details(window, cx)
                        .ruler_shaper(&snapshot, DisplayRow(0)),
                );
                let fresh = fresh_ruler(&snapshot, DisplayRow(0), &shaper);
                let cached = snapshot
                    .ruled_row(DisplayRow(0), shaper.clone())
                    .unwrap()
                    .ruler;
                (cached, fresh, shaper)
            })
        };
        let red = HighlightStyle {
            color: Some(gpui::red()),
            ..HighlightStyle::default()
        };
        let explicit_normal_red = HighlightStyle {
            font_weight: Some(FontWeight::NORMAL),
            ..red
        };
        let blue = HighlightStyle {
            color: Some(gpui::blue()),
            ..HighlightStyle::default()
        };
        let (letters, words) = (
            HighlightKey::DocumentHighlightRead,
            HighlightKey::DocumentHighlightWrite,
        );

        highlight(&mut cx, letters, 0, explicit_normal_red);
        highlight(&mut cx, words, 1, blue);
        let (separate, ..) = rulers(&mut cx);
        highlight(&mut cx, words, 1, red);
        let (cached, fresh, shaper) = rulers(&mut cx);

        assert!(fresh.width() < separate.width());
        assert!(
            (cached.width() - fresh.width()).abs() < 0.1,
            "{} != {}",
            cached.width(),
            fresh.width()
        );
        let with_syntax = |capture_name| {
            let mut themed = shaper.clone();
            themed.style.syntax =
                Arc::new(theme::SyntaxTheme::new_test([(capture_name, gpui::red())]));
            themed.metrics_fingerprint()
        };
        assert_ne!(with_syntax("comment"), with_syntax("comment.doc"));
        let mut recolored = shaper;
        recolored.style.status.warning = gpui::green();
        recolored.style.background = gpui::green();
        assert!(!recolored.matches(&cached));
    }

    #[gpui::test]
    async fn test_rulers_follow_syntax_changes_outside_the_edited_range(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let language = language::rust_lang();
        language.set_theme(&theme::SyntaxTheme::new_test([("comment", gpui::red())]));
        cx.set_state(&format!("ˇ/*\n{}\n*/\nfn main() {{}}", "x".repeat(4_096)));
        cx.update_buffer(|buffer, cx| buffer.set_language(Some(language), cx));
        cx.run_until_parked();
        let commented = displayed_row_ruler(&mut cx, 1);
        assert!(commented.syntax().is_some());
        cx.update_editor(|editor, _, cx| {
            editor.edit([(Point::new(0, 0)..Point::new(0, 2), "")], cx)
        });
        cx.run_until_parked();
        assert!(!cx.buffer(|buffer, _| buffer.is_parsing()));
        let uncommented = displayed_row_ruler(&mut cx, 1);
        assert!(!Arc::ptr_eq(&commented, &uncommented));
        assert_eq!(commented.starts, uncommented.starts);
    }

    #[test]
    fn test_grapheme_boundary_search_is_bounded_by_the_limit() {
        let text = format!("a{}b", "\u{301}".repeat(100));
        let cluster_end = text.len() as u32 - 1;
        let at_or_after = |boundaries: &GraphemeBoundaries, offset, limit| {
            boundaries.next_matching(offset, limit, |_| true)
        };
        let boundaries = GraphemeBoundaries::new(&text);
        assert_eq!(at_or_after(&boundaries, 1, 50), None);
        assert_eq!(at_or_after(&boundaries, 1, cluster_end), None);
        assert_eq!(
            at_or_after(&boundaries, 1, text.len() as u32),
            Some(cluster_end)
        );
        assert_eq!(
            at_or_after(&boundaries, cluster_end, text.len() as u32),
            Some(cluster_end)
        );
        assert!(boundaries.is_boundary(cluster_end));
        assert!(!boundaries.is_boundary(3));

        let flags = "🇺🇸".repeat(10);
        let boundaries = GraphemeBoundaries::new(&flags);
        assert_eq!(at_or_after(&boundaries, 1, 7), None);
        assert_eq!(at_or_after(&boundaries, 1, 8), Some(8));
        assert_eq!(at_or_after(&boundaries, 9, flags.len() as u32), Some(16));

        for len in [63, 127, 6207] {
            let text = format!("🇺🇸a{}", "b".repeat(len - "🇺🇸a".len()));
            let len = len as u32;
            let boundaries = GraphemeBoundaries::new(&text);
            assert_eq!(boundaries.next_matching(len - 1, len, |_| false), None);
            assert_eq!(
                boundaries.next_matching(len - 1, len, |offset| offset == len),
                Some(len)
            );
        }
    }

    #[gpui::test]
    async fn test_ruler_bounds_chunks_across_oversized_grapheme_clusters(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let cluster = format!("a{}", "\u{301}".repeat(20_000));
        let text = format!("{}{cluster}{}", "x".repeat(3_000), "y".repeat(3_000));
        let (ruler, ..) = ruler_for_text(&mut cx, &text);
        assert_eq!(ruler.len(), text.len() as u32);
        let boundaries = GraphemeBoundaries::new(&text);
        let cluster_range = 3_000..3_000 + cluster.len() as u32;
        let mut split_cluster = false;
        for (ix, boundary) in ruler.starts.iter().enumerate() {
            assert!(text.is_char_boundary(*boundary as usize));
            if cluster_range.start < *boundary && *boundary < cluster_range.end {
                split_cluster = true;
                assert!(
                    ruler.chunks[ix - 1].context_after > 0 && ruler.chunks[ix].context_before > 0,
                    "{boundary} drops shaping context inside the cluster"
                );
            } else {
                assert!(
                    boundaries.is_boundary(*boundary),
                    "{boundary} splits a cluster"
                );
            }
        }
        assert!(split_cluster);
        for chunk in &ruler.chunks {
            assert!(chunk.len + chunk.context_before + chunk.context_after <= MAX_SHAPED_LEN);
        }
    }

    #[gpui::test]
    async fn test_ruler_chunks_stay_bounded_on_replacement_heavy_rows(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let pairs = 8 * CHUNK_LEN;
        let text = "\u{1}a".repeat(pairs as usize);
        let (ruler, snapshot, shaper) = ruler_for_text(&mut cx, &text);
        assert_eq!(ruler.len(), text.len() as u32);
        assert!(ruler.chunks.len() <= (text.len() / CHUNK_LEN as usize) + 1);

        let symbol = crate::display_map::replacement('\u{1}')
            .map(String::from)
            .unwrap_or_default();
        let width_of = |text: &str| shaper.layout(text, &[shaper.run(None, text.len())]).width;
        let pair_width = ScrollPixelOffset::from(width_of(&symbol) + width_of("a"));
        let tolerance = pair_width * pairs as ScrollPixelOffset * 1e-4;
        for pair in [0, 1, CHUNK_LEN / 2 + 1, pairs - 1, pairs] {
            let x = ruler.x_for_column(2 * pair, &snapshot, DisplayRow(0), &shaper);
            let expected = pair_width * pair as ScrollPixelOffset;
            assert!(
                (x - expected).abs() < tolerance,
                "pair {pair}: {x} != {expected}"
            );
            assert_eq!(
                ruler.column_for_x(x, &snapshot, DisplayRow(0), &shaper),
                2 * pair
            );
        }
    }

    #[test]
    fn test_render_pieces_shape_visible_then_densest_pieces_within_budget() {
        let chunk = |width: f32| RulerChunk {
            width: px(width),
            measured: true,
            ..RulerChunk::unmeasured(CHUNK_LEN)
        };
        let chunks = iter::once(chunk(10.))
            .chain(iter::repeat_n(chunk(0.), 300))
            .chain([chunk(10.), chunk(1_000.)])
            .collect::<Vec<_>>();
        let ruler = RowRuler::assemble(0, None, Vec::new(), chunks);
        let shaped = |visible_x: Range<ScrollPixelOffset>| {
            ruler
                .render_pieces(0..ruler.len(), visible_x)
                .iter()
                .enumerate()
                .filter(|(_, piece)| piece.shaped)
                .map(|(ix, _)| ix)
                .collect::<Vec<_>>()
        };
        let budget_pieces = (MAX_RENDERED_LEN / CHUNK_LEN) as usize;
        let expected = (0..budget_pieces - 2).chain([301, 302]).collect::<Vec<_>>();
        assert_eq!(shaped(0.0..5.0), expected);
        assert_eq!(shaped(0.0..2_000.0), expected);
    }

    #[gpui::test(iterations = 100)]
    fn test_composed_row_edits_match_sequential_edits(mut rng: StdRng) {
        let original = (0..64).map(Some).collect::<Vec<_>>();
        let mut sequential = original.clone();
        let mut patches = Vec::new();
        for _ in 0..rng.random_range(1..8) {
            let mut edits = Vec::new();
            let mut column = 0;
            while column <= sequential.len() && edits.len() < 4 {
                let start = rng.random_range(column..=sequential.len());
                let end = rng.random_range(start..=sequential.len().min(start + 4));
                edits.push(RowEdit {
                    old: start as u32..end as u32,
                    new_len: rng.random_range(0..4),
                });
                column = if start < end { end } else { end + 1 };
            }
            apply_row_edits(&mut sequential, &edits);
            patches.push(edits);
        }
        let composed =
            compose_row_edits(patches.iter().map(Vec::as_slice)).expect("valid patches compose");
        let mut applied = original;
        apply_row_edits(&mut applied, &composed);
        assert_eq!(
            applied, sequential,
            "{patches:?} composed into {composed:?}"
        );
    }

    #[gpui::test(iterations = 20)]
    async fn test_cached_rulers_match_fresh_rulers_after_random_changes(
        cx: &mut TestAppContext,
        mut rng: StdRng,
    ) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let alphabet = ["漢", "a", "b", " ", "\t", "\u{1}", "é", "\n"];
        let random_text = |rng: &mut StdRng, len: usize, newlines: bool| {
            let choices = if newlines {
                &alphabet[..]
            } else {
                &alphabet[..alphabet.len() - 1]
            };
            (0..len)
                .map(|_| choices[rng.random_range(0..choices.len())])
                .collect::<String>()
        };
        let rows = (0..4)
            .map(|_| random_text(&mut rng, 700, false))
            .collect::<Vec<_>>();
        cx.set_state(&format!("ˇ{}", rows.join("\n")));

        let mut inlay_ids = Vec::new();
        let mut next_inlay_id = 0;
        let mut retained_count = 0;
        for _ in 0..20 {
            let before = long_row_rulers(&mut cx);
            let operation = rng.random_range(0..10);
            cx.update_editor(|editor, window, cx| {
                let buffer = editor.buffer().read(cx).snapshot(cx);
                let random_offset = |rng: &mut StdRng| {
                    buffer.clip_offset(
                        rng.random_range(MultiBufferOffset(0)..=buffer.len()),
                        Bias::Left,
                    )
                };
                match operation {
                    0..=4 => {
                        let start = random_offset(&mut rng);
                        let end = buffer.clip_offset(
                            (start + rng.random_range(0..4usize)).min(buffer.len()),
                            Bias::Right,
                        );
                        let len = rng.random_range(0..4);
                        let text = random_text(&mut rng, len, true);
                        editor.edit([(start..end, text)], cx);
                    }
                    5..=6 => {
                        let offset = random_offset(&mut rng);
                        let position = if rng.random_bool(0.5) {
                            buffer.anchor_before(offset)
                        } else {
                            buffer.anchor_after(offset)
                        };
                        let id = next_inlay_id;
                        next_inlay_id += 1;
                        inlay_ids.push(id);
                        let len = rng.random_range(1..4);
                        let text = random_text(&mut rng, len, false);
                        let inlay = Inlay::mock_hint(id, position, text.as_str());
                        editor
                            .display_map
                            .update(cx, |map, cx| map.splice_inlays(&[], vec![inlay], cx));
                    }
                    7 => {
                        if !inlay_ids.is_empty() {
                            let id = inlay_ids.swap_remove(rng.random_range(0..inlay_ids.len()));
                            editor.display_map.update(cx, |map, cx| {
                                map.splice_inlays(&[InlayId::Hint(id)], Vec::new(), cx)
                            });
                        }
                    }
                    8 => {
                        let start = random_offset(&mut rng);
                        let end = buffer.clip_offset(
                            (start + rng.random_range(1..20usize)).min(buffer.len()),
                            Bias::Right,
                        );
                        editor.fold_ranges(vec![start..end], false, window, cx);
                    }
                    _ => {
                        editor.unfold_ranges(
                            &[MultiBufferOffset(0)..buffer.len()],
                            true,
                            false,
                            cx,
                        );
                    }
                }
            });

            for (row, cached, fresh) in long_row_rulers(&mut cx) {
                retained_count += before
                    .iter()
                    .filter(|(_, previous, _)| Arc::ptr_eq(previous, &cached))
                    .count();
                assert_eq!(
                    cached.len(),
                    fresh.len(),
                    "row {row:?} after operation {operation}"
                );
                assert!(
                    (cached.width() - fresh.width()).abs() <= fresh.width() * 1e-5,
                    "row {row:?} after operation {operation}: {} != {}",
                    cached.width(),
                    fresh.width()
                );
            }
        }
        assert!(retained_count > 0);
    }

    #[gpui::test]
    async fn test_rulers_rebuild_when_boundary_edits_move_edge_inlays(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let long = "漢".repeat(MAX_LINE_LEN);
        for (id, inlay_at_row_end) in [(0, true), (1, false)] {
            let mut cx = EditorTestContext::new(cx).await;
            cx.set_state(&format!("ˇ{long}\nz"));
            cx.update_editor(|editor, _, cx| {
                let buffer = editor.buffer().read(cx).snapshot(cx);
                let position = if inlay_at_row_end {
                    buffer.anchor_after(MultiBufferOffset(long.len()))
                } else {
                    buffer.anchor_before(MultiBufferOffset(0))
                };
                let inlay = Inlay::mock_hint(id, position, "x");
                editor
                    .display_map
                    .update(cx, |map, cx| map.splice_inlays(&[], vec![inlay], cx));
            });
            let with_inlay = displayed_row_ruler(&mut cx, 0);
            assert_eq!(with_inlay.len(), long.len() as u32 + 1);

            let (at, inserted, long_row) = if inlay_at_row_end {
                (Point::new(0, long.len() as u32), "\n", 0)
            } else {
                (Point::zero(), "y\n", 1)
            };
            insert(&mut cx, at, inserted);
            let without_inlay = displayed_row_ruler(&mut cx, long_row);
            assert!(!Arc::ptr_eq(&with_inlay, &without_inlay));
            assert_eq!(without_inlay.len(), long.len() as u32);
        }
    }

    #[gpui::test]
    async fn test_multiline_inlay_highlights_rebuild_continuation_rows(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        cx.set_state("ˇab");
        let inlay_text = format!("p\n{}\nr", "漢".repeat(MAX_LINE_LEN));
        let inlay_position = cx.update_editor(|editor, _, cx| {
            let position = editor
                .buffer()
                .read(cx)
                .snapshot(cx)
                .anchor_after(MultiBufferOffset(1));
            let inlay = Inlay::mock_hint(0, position, inlay_text.as_str());
            editor
                .display_map
                .update(cx, |map, cx| map.splice_inlays(&[], vec![inlay], cx));
            position
        });
        let before = displayed_row_ruler(&mut cx, 1);
        assert_eq!(before.len(), (inlay_text.len() - "p\n\nr".len()) as u32);

        highlight_first_inlay(&mut cx, inlay_position, 2..5);
        assert!(!Arc::ptr_eq(&before, &displayed_row_ruler(&mut cx, 1)));
    }

    #[gpui::test]
    async fn test_inlay_highlights_rebuild_rows_shifted_by_co_anchored_multiline_inlays(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        cx.set_state("ˇab");
        let long_hint = "漢".repeat(MAX_LINE_LEN);
        let inlay_position = cx.update_editor(|editor, _, cx| {
            let position = editor
                .buffer()
                .read(cx)
                .snapshot(cx)
                .anchor_before(MultiBufferOffset(1));
            let inlays = vec![
                Inlay::mock_hint(0, position, long_hint.as_str()),
                Inlay::mock_hint(1, position, "\nz"),
            ];
            editor
                .display_map
                .update(cx, |map, cx| map.splice_inlays(&[], inlays, cx));
            position
        });
        let before = displayed_row_ruler(&mut cx, 0);
        assert_eq!(before.len(), ("a".len() + long_hint.len()) as u32);

        highlight_first_inlay(&mut cx, inlay_position, 0..3);
        assert!(!Arc::ptr_eq(&before, &displayed_row_ruler(&mut cx, 0)));
    }

    #[gpui::test]
    async fn test_rulers_above_sync_limit_build_in_background_and_follow_edits(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let long = "漢字".repeat(3_000);
        assert!(long.len() as u32 > MAX_SYNC_RULER_LEN);
        cx.set_state(&format!("ˇshort\n{long}\ntail"));
        let cached_and_fresh = |cx: &mut EditorTestContext, row: u32| {
            cx.update_editor(|editor, window, cx| {
                let snapshot = editor.snapshot(window, cx).display_snapshot;
                let shaper = editor
                    .text_layout_details(window, cx)
                    .ruler_shaper(&snapshot, DisplayRow(row));
                let fresh = fresh_ruler(&snapshot, DisplayRow(row), &shaper);
                let cached = snapshot
                    .ruled_row(DisplayRow(row), shaper)
                    .map(|ruled| ruled.ruler);
                (cached, fresh)
            })
        };
        let assert_matches_fresh = |(cached, fresh): (Option<Arc<RowRuler>>, RowRuler)| {
            let cached = cached.expect("a ruler for the long row");
            assert!(!cached.needs_measuring());
            assert_eq!(cached.len(), fresh.len());
            assert!(
                (cached.width() - fresh.width()).abs() <= fresh.width() * 1e-5,
                "{} != {}",
                cached.width(),
                fresh.width()
            );
        };

        assert!(cached_and_fresh(&mut cx, 1).0.is_none());
        cx.update_editor(|editor, _, cx| {
            editor
                .display_map
                .update(cx, |map, cx| map.spawn_ruler_builds(None, cx))
        });
        insert(&mut cx, Point::new(0, 0), "above\n");
        insert(&mut cx, Point::new(2, 30), "漢");
        assert!(cached_and_fresh(&mut cx, 2).0.is_none());
        cx.run_until_parked();
        assert_matches_fresh(cached_and_fresh(&mut cx, 2));

        insert(&mut cx, Point::new(2, 9_000), "漢a");
        assert_matches_fresh(cached_and_fresh(&mut cx, 2));

        cx.update_editor(|editor, _, cx| {
            let at = |column| Point::new(2, column);
            editor.edit(
                [
                    (at(3)..at(3), "b"),
                    (at(12)..at(18), ""),
                    (at(12_001)..at(12_001), "漢 "),
                ],
                cx,
            )
        });
        assert_matches_fresh(cached_and_fresh(&mut cx, 2));

        insert(&mut cx, Point::new(2, 3), &"漢".repeat(6_000));
        assert!(cached_and_fresh(&mut cx, 2).0.is_none());
        build_queued_rulers(&mut cx);
        assert_matches_fresh(cached_and_fresh(&mut cx, 2));

        let unmeasured_len = cx.update_editor(|editor, window, cx| {
            let row_text = editor
                .buffer()
                .read(cx)
                .snapshot(cx)
                .text_for_range(Point::new(2, 0)..Point::new(3, 0))
                .collect::<String>();
            let edits = (1..=9)
                .map(|ix| {
                    let column = (0..=ix * 3_500)
                        .rev()
                        .find(|column| row_text.is_char_boundary(*column))
                        .unwrap_or(0) as u32;
                    (Point::new(2, column)..Point::new(2, column), "漢")
                })
                .collect::<Vec<_>>();
            editor.edit(edits, cx);
            let snapshot = editor.snapshot(window, cx).display_snapshot;
            let shaper = editor
                .text_layout_details(window, cx)
                .ruler_shaper(&snapshot, DisplayRow(2));
            let tab_row = snapshot.unwrapped_tab_row(DisplayRow(2));
            let unmeasured_len = snapshot
                .row_rulers
                .current(tab_row, &shaper)
                .map(|(ruler, _)| ruler.unmeasured_len());
            assert!(snapshot.ruled_row(DisplayRow(2), shaper).is_none());
            unmeasured_len
        });
        assert!(unmeasured_len.is_some_and(|len| len > MAX_SYNC_RULER_LEN));
        build_queued_rulers(&mut cx);
        assert_matches_fresh(cached_and_fresh(&mut cx, 2));
    }

    #[gpui::test]
    async fn test_unmeasured_rows_map_coordinates_exactly(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        let long = "é".repeat(10_000);
        assert!(long.len() as u32 > MAX_SYNC_RULER_LEN);
        cx.set_state(&format!("ˇshort\n{long}"));
        cx.update_editor(|editor, window, cx| {
            let snapshot = editor.snapshot(window, cx).display_snapshot;
            let details = editor.text_layout_details(window, cx);
            let row = DisplayRow(1);
            let shaper = details.ruler_shaper(&snapshot, row);
            let tab_row = snapshot.unwrapped_tab_row(row);
            assert_eq!(
                snapshot.display_column_for_x(row, f64::INFINITY, &details),
                long.len() as u32
            );
            assert!(snapshot.row_rulers.prefix(tab_row, &shaper).is_none());
            let measured = fresh_ruler(&snapshot, row, &shaper);
            for column in [200, 9_000, 19_000] {
                let x = measured.x_for_column(column, &snapshot, row, &shaper);
                let mapped_x =
                    snapshot.x_for_display_point(DisplayPoint::new(row, column), &details);
                assert_eq!(snapshot.display_column_for_x(row, x, &details), column);
                assert!((mapped_x - x).abs() < 0.01, "{mapped_x} != {x}");
            }
            assert!(snapshot.row_rulers.current(tab_row, &shaper).is_none());
        });
    }

    #[gpui::test]
    async fn test_unmeasured_windows_split_inlays_at_char_boundaries(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let long = "é".repeat(10_000);
        for (inlay_text, row, expected) in [(format!("\n{long}\nr"), 1, 0..656), (long, 0, 0..655)]
        {
            let mut cx = EditorTestContext::new(cx).await;
            cx.set_state("ˇab");
            cx.update_editor(|editor, window, cx| {
                let position = editor
                    .buffer()
                    .read(cx)
                    .snapshot(cx)
                    .anchor_after(MultiBufferOffset(1));
                let inlay = Inlay::mock_hint(0, position, inlay_text.as_str());
                editor
                    .display_map
                    .update(cx, |map, cx| map.splice_inlays(&[], vec![inlay], cx));
                let snapshot = editor.snapshot(window, cx).display_snapshot;
                let cell = editor.text_layout_details(window, cx).grid_cell(&snapshot);
                let row = DisplayRow(row);
                let viewport = HorizontalViewport {
                    scroll_columns: 0.,
                    visible_columns: 100.,
                    text_align: TextAlign::Left,
                    content_width: px(1_000.),
                };
                let estimated = EstimatedRow::new(&snapshot, row, snapshot.line_len(row));
                assert_eq!(
                    snapshot.estimated_window(&estimated, &viewport, cell),
                    expected
                );
            });
        }
    }

    #[gpui::test]
    async fn test_queued_ruler_requests_use_current_widths_and_die_with_their_cache(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let (buffer, map) = long_row_test_map(&"漢字".repeat(3_000), cx);
        let shaper = test_shaper(cx);
        let row = DisplayRow(0);
        map.update(cx, |map, cx| {
            let position = buffer
                .read(cx)
                .snapshot(cx)
                .anchor_before(MultiBufferOffset(3));
            let swatch = Inlay::color(0, position, gpui::Rgba::default());
            map.splice_inlays(&[], vec![swatch], cx);
        });
        let queue_request = |cx: &mut TestAppContext| {
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            assert!(snapshot.ruled_row(row, shaper.clone()).is_none());
            Arc::downgrade(&snapshot.row_rulers)
        };

        let abandoned_cache = queue_request(cx);
        buffer.update(cx, |buffer, cx| {
            buffer.edit([(Point::zero()..Point::zero(), "x")], None, cx)
        });
        queue_request(cx);
        assert!(abandoned_cache.upgrade().is_none());

        let swatch_width = px(40.);
        map.update(cx, |map, cx| {
            let widths = [(ChunkRendererId::Inlay(InlayId::Color(0)), swatch_width)];
            assert!(map.update_fold_widths(widths, shaper.renderer_metrics_key(), cx));
        });
        queue_request(cx);
        map.update(cx, |map, cx| map.spawn_ruler_builds(None, cx));
        cx.run_until_parked();

        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let renderer_widths = [(InlayId::Color(0), swatch_width)]
            .into_iter()
            .collect::<HashMap<_, _>>();
        let expected = RowRuler::new(&snapshot, row, &shaper, None, &renderer_widths);
        let unmeasured = fresh_ruler(&snapshot, row, &shaper);
        assert_ne!(expected.width(), unmeasured.width());
        let built = snapshot.ruled_row(row, shaper.clone()).unwrap().ruler;
        assert_eq!(built.width(), expected.width());
    }

    #[gpui::test]
    async fn test_trailing_whitespace_follows_edits_without_rescanning(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let (buffer, map) = long_row_test_map("hello world   \nnext", cx);
        let trailing_start = |computed: Option<usize>, cx: &mut TestAppContext| {
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            snapshot
                .row_rulers
                .trailing_whitespace_start(0, snapshot.line_len(DisplayRow(0)), || {
                    TrailingWhitespace {
                        row_len: snapshot.line_len(DisplayRow(0)),
                        start: computed.expect("a cached trailing whitespace start"),
                        exact: true,
                        idle_refreshes: 0,
                    }
                })
                .start
        };
        let edit = |range: Range<Point>, text: &str, cx: &mut TestAppContext| {
            buffer.update(cx, |buffer, cx| buffer.edit([(range, text)], None, cx));
        };

        assert_eq!(trailing_start(Some(11), cx), 11);
        edit(Point::new(0, 0)..Point::new(0, 0), ">> ", cx);
        assert_eq!(trailing_start(None, cx), 14);
        edit(Point::new(0, 15)..Point::new(0, 16), "", cx);
        assert_eq!(trailing_start(None, cx), 14);
        edit(Point::new(1, 0)..Point::new(1, 0), "x", cx);
        assert_eq!(trailing_start(None, cx), 14);
        edit(Point::new(0, 15)..Point::new(0, 15), "!", cx);
        assert_eq!(trailing_start(None, cx), 16);
        edit(Point::new(0, 17)..Point::new(0, 17), "  ", cx);
        assert_eq!(trailing_start(None, cx), 16);
    }

    #[gpui::test]
    async fn test_long_trailing_whitespace_is_scanned_in_background(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let text = format!("x{}", " ".repeat(4 * MAX_SYNC_RULER_LEN as usize));
        let (_, map) = long_row_test_map(&text, cx);
        let trailing_start = |cx: &mut TestAppContext| {
            map.update(cx, |map, cx| {
                map.snapshot(cx)
                    .cached_trailing_whitespace_start(DisplayRow(0))
            })
        };

        assert_eq!(trailing_start(cx), text.len() - MAX_SYNC_RULER_LEN as usize);
        map.update(cx, |map, cx| map.spawn_ruler_builds(None, cx));
        cx.run_until_parked();
        assert_eq!(trailing_start(cx), 1);
    }

    #[gpui::test]
    async fn test_grid_mismatch_moves_font_to_rulers(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        for text in ["VA", "->"] {
            let mut cx = EditorTestContext::new(cx).await;
            cx.set_state(&format!("ˇ{}", text.repeat(MAX_LINE_LEN)));
            cx.update_editor(|editor, window, cx| {
                let snapshot = editor.snapshot(window, cx).display_snapshot;
                let details = editor.text_layout_details(window, cx);
                let measure = |snapshot: &DisplaySnapshot| {
                    GridCell::measure(
                        &details.text_system,
                        &details.editor_style,
                        snapshot,
                        details.font_size(),
                    )
                };
                let row = DisplayRow(0);
                let cell = measure(&snapshot);
                assert!(cell.monospace, "{text}");
                let shaper = contextual_shaper(details.ruler_shaper(&snapshot, row));
                let viewport = details.horizontal_viewport(&snapshot);
                let row_len = snapshot.line_len(row);
                assert!(
                    snapshot
                        .grid_window(row, row_len, &viewport, cell, &shaper)
                        .is_none(),
                    "{text}"
                );
                assert!(!measure(&snapshot).monospace, "{text}");

                editor.edit([(Point::zero()..Point::zero(), "V")], cx);
                assert!(
                    !measure(&editor.snapshot(window, cx).display_snapshot).monospace,
                    "{text}"
                );
            });
        }
    }

    #[gpui::test]
    async fn test_enabled_font_features_disable_the_grid(cx: &mut TestAppContext) {
        cx.update(|cx| init_test(cx, &|_| {}));
        let mut cx = EditorTestContext::new(cx).await;
        cx.set_state("ˇx");
        cx.update_editor(|editor, window, cx| {
            let snapshot = editor.snapshot(window, cx).display_snapshot;
            let details = editor.text_layout_details(window, cx);
            let monospace = |features: &[(&str, u32)]| {
                let mut style = details.editor_style.clone();
                style.text.font_features = FontFeatures(Arc::new(
                    features
                        .iter()
                        .map(|(tag, value)| (tag.to_string(), *value))
                        .collect(),
                ));
                GridCell::measure(&details.text_system, &style, &snapshot, details.font_size())
                    .monospace
            };
            assert_eq!(
                (
                    monospace(&[]),
                    monospace(&[("calt", 0)]),
                    monospace(&[("frac", 1)])
                ),
                (true, true, false)
            );
        });
    }

    fn apply_row_edits(text: &mut Vec<Option<u32>>, edits: &[RowEdit]) {
        for edit in edits.iter().rev() {
            text.splice(
                edit.old.start as usize..edit.old.end as usize,
                iter::repeat_n(None, edit.new_len as usize),
            );
        }
    }

    fn ruler_for_text(
        cx: &mut EditorTestContext,
        text: &str,
    ) -> (RowRuler, DisplaySnapshot, RulerShaper) {
        let (snapshot, shaper) = snapshot_for_text(cx, text);
        let ruler = fresh_ruler(&snapshot, DisplayRow(0), &shaper);
        (ruler, snapshot, shaper)
    }

    fn snapshot_for_text(cx: &mut EditorTestContext, text: &str) -> (DisplaySnapshot, RulerShaper) {
        cx.set_state(&format!("ˇ{text}"));
        cx.update_editor(|editor, window, cx| {
            let snapshot = editor.snapshot(window, cx).display_snapshot;
            let shaper = editor
                .text_layout_details(window, cx)
                .ruler_shaper(&snapshot, DisplayRow(0));
            (snapshot, shaper)
        })
    }

    fn fresh_ruler(snapshot: &DisplaySnapshot, row: DisplayRow, shaper: &RulerShaper) -> RowRuler {
        RowRuler::new(snapshot, row, shaper, None, &HashMap::default())
    }

    fn insert(cx: &mut EditorTestContext, at: Point, text: &str) {
        cx.update_editor(|editor, _, cx| editor.edit([(at..at, text)], cx));
    }

    fn build_queued_rulers(cx: &mut EditorTestContext) {
        cx.update_editor(|editor, _, cx| {
            editor
                .display_map
                .update(cx, |map, cx| map.spawn_ruler_builds(None, cx))
        });
        cx.run_until_parked();
    }

    fn bold() -> HighlightStyle {
        HighlightStyle {
            font_weight: Some(FontWeight::BOLD),
            ..HighlightStyle::default()
        }
    }

    fn highlight_first_inlay(
        cx: &mut EditorTestContext,
        inlay_position: Anchor,
        range: Range<usize>,
    ) {
        cx.update_editor(|editor, _, cx| {
            editor.highlight_inlays(
                HighlightKey::HoveredLinkState,
                vec![InlayHighlight {
                    inlay: InlayId::Hint(0),
                    inlay_position,
                    range,
                }],
                bold(),
                cx,
            )
        });
    }

    fn long_row_rulers(cx: &mut EditorTestContext) -> Vec<(DisplayRow, Arc<RowRuler>, RowRuler)> {
        cx.update_editor(|editor, window, cx| {
            let snapshot = editor.snapshot(window, cx).display_snapshot;
            let details = editor.text_layout_details(window, cx);
            (0..=snapshot.max_point().row().0)
                .map(DisplayRow)
                .filter(|row| snapshot.is_long_unwrapped_row(*row))
                .map(|row| {
                    let shaper = details.ruler_shaper(&snapshot, row);
                    let fresh = fresh_ruler(&snapshot, row, &shaper);
                    (row, snapshot.ruled_row(row, shaper).unwrap().ruler, fresh)
                })
                .collect()
        })
    }

    fn contextual_shaper(shaper: RulerShaper) -> RulerShaper {
        let text_system = TextSystem::new(Arc::new(ContextualTextSystem(NoopTextSystem)));
        RulerShaper {
            text_system: Arc::new(WindowTextSystem::new(Arc::new(text_system))),
            ..shaper
        }
    }

    struct ContextualTextSystem(NoopTextSystem);

    impl PlatformTextSystem for ContextualTextSystem {
        fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> anyhow::Result<()> {
            self.0.add_fonts(fonts)
        }

        fn all_font_names(&self) -> Vec<String> {
            self.0.all_font_names()
        }

        fn font_id(&self, descriptor: &Font) -> anyhow::Result<FontId> {
            self.0.font_id(descriptor)
        }

        fn font_metrics(&self, font_id: FontId) -> FontMetrics {
            self.0.font_metrics(font_id)
        }

        fn typographic_bounds(
            &self,
            font_id: FontId,
            glyph_id: GlyphId,
        ) -> anyhow::Result<Bounds<f32>> {
            self.0.typographic_bounds(font_id, glyph_id)
        }

        fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> anyhow::Result<Size<f32>> {
            self.0.advance(font_id, glyph_id)
        }

        fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
            self.0.glyph_for_char(font_id, ch)
        }

        fn glyph_raster_bounds(
            &self,
            params: &RenderGlyphParams,
        ) -> anyhow::Result<Bounds<DevicePixels>> {
            self.0.glyph_raster_bounds(params)
        }

        fn rasterize_glyph(
            &self,
            params: &RenderGlyphParams,
            raster_bounds: Bounds<DevicePixels>,
        ) -> anyhow::Result<(Size<DevicePixels>, Vec<u8>)> {
            self.0.rasterize_glyph(params, raster_bounds)
        }

        fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
            assert!(text.encode_utf16().count() <= 10_240);
            let em = f64::from(f32::from(font_size)) * 0.6;
            let mut x = 0.;
            let mut run_start = 0;
            let mut shaped_runs = Vec::new();
            for run in runs {
                let mut glyphs = Vec::new();
                let mut previous = None;
                let mut denominator = false;
                let mut characters = text[run_start..run_start + run.len]
                    .char_indices()
                    .peekable();
                while let Some((index, character)) = characters.next() {
                    if previous == Some('V') && character == 'A' {
                        x -= em / 4.;
                    }
                    glyphs.push(ShapedGlyph {
                        id: GlyphId(u32::from(character)),
                        position: point(px(x as f32), px(0.)),
                        index: run_start + index,
                        is_emoji: false,
                    });
                    previous = Some(character);
                    denominator = character == '/' || denominator && character.is_ascii_digit();
                    if character == '-'
                        && let Some((_, arrow)) = characters.next_if(|(_, next)| *next == '>')
                    {
                        x += em;
                        glyphs.push(ShapedGlyph {
                            id: GlyphId(u32::from(arrow)),
                            position: point(px(x as f32), px(0.)),
                            index: run_start + index,
                            is_emoji: false,
                        });
                        previous = Some(arrow);
                        x += em;
                    } else if character == 'f'
                        && characters.next_if(|(_, next)| *next == 'i').is_some()
                    {
                        x += 1.5 * em;
                        previous = Some('i');
                    } else if denominator && character.is_ascii_digit() {
                        x += em / 2.;
                    } else {
                        x += em;
                    }
                }
                shaped_runs.push(ShapedRun {
                    font_id: run.font_id,
                    glyphs,
                });
                run_start += run.len;
            }
            let metrics = self.0.font_metrics(FontId(0));
            let units_per_em = metrics.units_per_em as f32;
            LineLayout {
                font_size,
                width: px(x as f32),
                ascent: font_size * (metrics.ascent / units_per_em),
                descent: font_size * (metrics.descent / units_per_em),
                runs: shaped_runs,
                len: text.len(),
            }
        }

        fn recommended_rendering_mode(
            &self,
            font_id: FontId,
            font_size: Pixels,
        ) -> TextRenderingMode {
            self.0.recommended_rendering_mode(font_id, font_size)
        }
    }
}
