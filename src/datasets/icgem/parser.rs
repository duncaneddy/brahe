//! HTML scraping for ICGEM listing pages.

use crate::datasets::icgem::body::ICGEMBody;
use crate::datasets::icgem::index::IndexEntry;
use crate::utils::BraheError;
use scraper::{Html, Selector};

/// Collect the text content of an element, inserting a space at every `<br>`
/// boundary so that values separated only by `<br>` (with no surrounding
/// whitespace) become distinct whitespace-delimited tokens.
///
/// `scraper`'s `.text()` iterator yields only text nodes; `<br>` elements
/// produce no text, so adjacent values like `11000<br>5500<br>760` would be
/// concatenated into `"110005500760"` by a plain `.collect::<String>()`.
///
/// Implementation: split `inner_html()` on `<br>` tags (any case/variant),
/// strip remaining tags from each chunk, then join with a space.
fn text_with_br_spaces(el: &scraper::ElementRef<'_>) -> String {
    let inner = el.inner_html();
    split_on_br(&inner)
        .into_iter()
        .map(strip_html_tags)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split a raw HTML string on `<br>`, `<br/>`, or `<br ...>` tags (any case).
fn split_on_br(html: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let bytes = html.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if bytes[i] == b'<' {
            // Peek at the tag name, skipping an optional leading '/'.
            let after_open = i + 1;
            let tag_start = if after_open < len && bytes[after_open] == b'/' {
                after_open + 1
            } else {
                after_open
            };
            if tag_start + 2 <= len {
                let tag_lo = html[tag_start..tag_start + 2].to_ascii_lowercase();
                if tag_lo == "br" {
                    // Find the closing '>'.
                    if let Some(close_offset) = html[i..].find('>') {
                        parts.push(&html[start..i]);
                        i += close_offset + 1; // advance past '>'
                        start = i;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    parts.push(&html[start..]);
    parts
}

/// Strip all `<...>` HTML tags from a string slice, returning plain text.
fn strip_html_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

/// Parse the Earth catalog HTML (`tom_longtime`) into index entries.
///
/// All entries are tagged with `ICGEMBody::Earth`.
pub fn parse_earth_catalog(html: &str) -> Result<Vec<IndexEntry>, BraheError> {
    parse_catalog(html, |_object_text| ICGEMBody::Earth)
}

/// Parse the celestial catalog HTML (`tom_celestial`) into index entries.
///
/// The body for each row is read from the "Object" column.
pub fn parse_celestial_catalog(html: &str) -> Result<Vec<IndexEntry>, BraheError> {
    parse_catalog(html, |object_text| {
        if object_text.is_empty() {
            ICGEMBody::Other("unknown".to_string())
        } else {
            ICGEMBody::from_name(object_text)
        }
    })
}

/// Parse any ICGEM listing-page HTML into `IndexEntry` values.
///
/// `classify_body` is called with the trimmed text of the "Object" column
/// (cell index 1) to determine the `ICGEMBody` for a row.
///
/// # Table layout (tom_longtime / tom_celestial)
///
/// Column indices (0-based `<td>` children of a data `<tr>`):
/// - 0  : Row number
/// - 1  : Object (body name; empty for Earth models)
/// - 2  : Model name (`fw-bold`, may contain an `<a>`)
/// - 3  : Year
/// - 4  : Degree (one or more values separated by `<br>`; first is primary)
/// - 5  : Data type codes
/// - 6  : References
/// - 7  : Download links (one or more `<a href="/getmodel/gfc/…">`)
/// - 8+ : Calculate / 3D / DOI (ignored)
///
/// One `IndexEntry` is emitted **per `getmodel/gfc/` link** found in cell 7.
/// Rows that have no such link (header rows, etc.) are skipped.
///
/// **Degree resolution:**
/// 1. Primary: cell[4] degree paired with download link by index. Multi-variant
///    rows list degrees separated by `<br>` (collapsed to whitespace by `.text()`);
///    the i-th degree corresponds to the i-th gfc link.
/// 2. Fallback: trailing `_NUMBER` suffix of the `.gfc` filename — used only
///    when cell[4] has fewer degree values than gfc links.
///
/// **Model name resolution:**
/// Model name comes from cell[2] (`.fw-bold`), which holds the canonical model
/// name as displayed by ICGEM (sometimes wrapped in an `<a>` tag; `.text()`
/// collects descendant text correctly). For multi-variant rows all emitted
/// entries share this name and differ only by `degree`.
fn parse_catalog<F>(html: &str, classify_body: F) -> Result<Vec<IndexEntry>, BraheError>
where
    F: Fn(&str) -> ICGEMBody,
{
    let doc = Html::parse_document(html);

    let row_sel = Selector::parse("tr").unwrap();
    let cell_sel = Selector::parse("td").unwrap();
    let link_sel = Selector::parse("a[href*='/getmodel/gfc/']").unwrap();

    let mut entries = Vec::new();

    for tr in doc.select(&row_sel) {
        // Collect all <td> children.
        let cells: Vec<_> = tr.select(&cell_sel).collect();
        // Must have at least 8 columns to contain a Download cell.
        if cells.len() < 8 {
            continue;
        }

        // Cell 7: Download — look for all gfc links.
        let download_cell = &cells[7];
        let gfc_links: Vec<_> = download_cell.select(&link_sel).collect();
        if gfc_links.is_empty() {
            continue;
        }

        // Cell 1: Object (body).
        let object_text = cells[1].text().collect::<String>();
        let object_text = object_text.trim();
        let body = classify_body(object_text);

        // Cell 2: Model name (.fw-bold cell; may contain an <a> wrapper).
        let name = cells[2].text().collect::<String>().trim().to_string();
        if name.is_empty() {
            continue;
        }

        // Cell 3: Year — first 4-digit integer in [1900, 2100].
        let year_text = cells[3].text().collect::<String>();
        let year = year_text.split_whitespace().find_map(|tok| {
            tok.parse::<u16>()
                .ok()
                .filter(|&y| (1900..=2100).contains(&y))
        });

        // Cell 4: All degrees in document order. Multi-variant rows list degrees
        // with `<br>` separators that `.text()` collapses to nothing — we use
        // `text_with_br_spaces` to insert a space at each `<br>` boundary so
        // that values like `11000<br>5500<br>760` become distinct tokens.
        let degree_text = text_with_br_spaces(&cells[4]);
        let cell_degrees: Vec<u32> = degree_text
            .split_whitespace()
            .filter_map(|tok| tok.parse::<u32>().ok())
            .filter(|&n| n >= 2)
            .collect();

        // Emit one IndexEntry per gfc link, paired positionally with cell_degrees.
        for (i, link) in gfc_links.iter().enumerate() {
            let href = match link.value().attr("href") {
                Some(h) => h.to_string(),
                None => continue,
            };

            // Primary degree source: the i-th value from the Degree cell.
            // Fallback: trailing `_NUMBER` in the filename (defensive — only used
            // if cell_degrees has fewer values than gfc_links).
            let degree = cell_degrees.get(i).copied().or_else(|| {
                let filename = href
                    .rsplit('/')
                    .next()
                    .unwrap_or("")
                    .trim_end_matches(".gfc");
                filename
                    .rfind('_')
                    .and_then(|pos| filename[pos + 1..].parse::<u32>().ok().filter(|&n| n >= 2))
            });

            let degree = match degree {
                Some(d) => d,
                None => continue,
            };

            entries.push(IndexEntry {
                body: body.clone(),
                name: name.clone(),
                year,
                degree,
                download_path: href,
            });
        }
    }

    if entries.is_empty() {
        return Err(BraheError::Error(
            "ICGEM catalog parse returned no entries — page format may have changed".to_string(),
        ));
    }

    Ok(entries)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    const EARTH_FIXTURE: &str = include_str!("../../../test_assets/icgem/tom_longtime_sample.html");

    #[test]
    fn test_parse_earth_catalog_has_entries() {
        let entries = parse_earth_catalog(EARTH_FIXTURE).unwrap();
        assert!(
            !entries.is_empty(),
            "expected at least 1 Earth model entry, got {}",
            entries.len()
        );
        for e in &entries {
            assert_eq!(e.body, ICGEMBody::Earth);
            assert!(!e.name.is_empty());
            assert!(e.download_path.contains("/getmodel/gfc/"));
            assert!(e.degree >= 2);
        }
    }

    #[test]
    fn test_parse_earth_catalog_empty_html_errors() {
        let result = parse_earth_catalog("<html><body></body></html>");
        assert!(result.is_err());
    }

    const CELESTIAL_FIXTURE: &str =
        include_str!("../../../test_assets/icgem/tom_celestial_sample.html");

    #[test]
    fn test_parse_earth_catalog_multi_variant_degrees_from_cell() {
        // WHU-CASM-UGM2025_2159 is in the fixture and has 4 download variants.
        // The Degree column lists 2190, 11000, 5500, 760 (in that order); the
        // filenames are _2159.gfc, _11000.gfc, _5399.gfc, _719.gfc. The parser
        // MUST use the Degree column, not the filename suffix.
        let entries = parse_earth_catalog(EARTH_FIXTURE).unwrap();
        let whu: Vec<&IndexEntry> = entries
            .iter()
            .filter(|e| e.name == "WHU-CASM-UGM2025_2159")
            .collect();
        assert_eq!(
            whu.len(),
            4,
            "expected 4 download variants for WHU-CASM-UGM2025_2159"
        );

        let mut degrees: Vec<u32> = whu.iter().map(|e| e.degree).collect();
        degrees.sort();
        assert_eq!(
            degrees,
            vec![760, 2190, 5500, 11000],
            "degrees must come from the Degree column, not the filename suffix"
        );
    }

    #[test]
    fn test_parse_earth_catalog_single_variant_uses_cell_name() {
        // EGM2008 is a single-variant row. Name comes from cell[2], degree from cell[4].
        let entries = parse_earth_catalog(EARTH_FIXTURE).unwrap();
        let egm = entries.iter().find(|e| e.name == "EGM2008");
        assert!(egm.is_some(), "EGM2008 entry missing");
        assert_eq!(egm.unwrap().degree, 2190);
    }

    #[test]
    fn test_parse_earth_catalog_keeps_year_range_degrees() {
        // EIGEN-6C2 and EIGEN-6C3stat have degree 1949, which falls within the
        // year range 1900-2100. The parser must not exclude it.
        let entries = parse_earth_catalog(EARTH_FIXTURE).unwrap();
        let degrees_1949: Vec<&IndexEntry> = entries.iter().filter(|e| e.degree == 1949).collect();
        assert!(
            !degrees_1949.is_empty(),
            "expected at least one entry with degree 1949 (e.g. EIGEN-6C2 or EIGEN-6C3stat)"
        );
    }

    #[test]
    fn test_parse_celestial_catalog_assigns_bodies() {
        let entries = parse_celestial_catalog(CELESTIAL_FIXTURE).unwrap();
        assert!(!entries.is_empty());

        // We expect Moon and Mars to appear among the rows; the fixture must include them.
        let bodies: std::collections::HashSet<_> = entries.iter().map(|e| e.body.clone()).collect();
        assert!(
            bodies.contains(&ICGEMBody::Moon) || bodies.contains(&ICGEMBody::Mars),
            "expected at least one Moon or Mars entry; got bodies: {:?}",
            bodies
        );

        // No row should be tagged as Earth on the celestial page.
        assert!(!bodies.contains(&ICGEMBody::Earth));
    }
}
