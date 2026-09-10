"use client";

import { useMemo, useState } from "react";
import { archiveItems, type ArchiveItem } from "../data/archive";

const filters = ["All", "Fort", "Battle", "Document", "Artefact"] as const;

export function ArchiveExplorer() {
  const [filter, setFilter] = useState<(typeof filters)[number]>("All");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<ArchiveItem | null>(null);
  const results = useMemo(() => archiveItems.filter((item) =>
    (filter === "All" || item.type === filter) &&
    `${item.title} ${item.place} ${item.description}`.toLowerCase().includes(query.toLowerCase())
  ), [filter, query]);

  return <section className="explorer section" id="explore" aria-labelledby="explore-title">
    <div className="eyebrow">The field archive</div>
    <div className="section-heading">
      <div><h2 id="explore-title">Explore the chronicles</h2><p>Trace landscapes, campaigns, courtly records, and the material culture of Swarajya.</p></div>
      <label className="search"><span aria-hidden="true">⌕</span><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search the archive" aria-label="Search the archive" /></label>
    </div>
    <div className="filter-row" role="tablist" aria-label="Archive categories">
      {filters.map((item) => <button key={item} role="tab" aria-selected={filter === item} className={filter === item ? "active" : ""} onClick={() => setFilter(item)}>{item}</button>)}
    </div>
    {results.length ? <div className="archive-grid">{results.map((item) => <article className="archive-card" key={item.title}>
      <div className="card-mark" aria-hidden="true">{item.symbol}</div><div className="card-meta"><span>{item.type}</span><span>{item.period}</span></div>
      <h3>{item.title}</h3><p className="place">{item.place}</p><p>{item.description}</p>
      <button className="text-button" onClick={() => setSelected(item)}>Open record <span>→</span></button>
    </article>)}</div> : <div className="empty"><span>⌕</span><h3>No record found</h3><p>Try another title, place, or category.</p><button onClick={() => { setFilter("All"); setQuery(""); }}>Clear filters</button></div>}
    {selected && <div className="modal-backdrop" role="presentation" onMouseDown={() => setSelected(null)}><section className="record-modal" role="dialog" aria-modal="true" aria-labelledby="record-title" onMouseDown={(e) => e.stopPropagation()}>
      <button className="close" onClick={() => setSelected(null)} aria-label="Close record">×</button><div className="card-mark">{selected.symbol}</div><div className="card-meta"><span>{selected.type}</span><span>{selected.period}</span></div>
      <h2 id="record-title">{selected.title}</h2><p className="place">{selected.place}</p><p>{selected.description}</p><hr/><p className="small">Archive note: this is curated demonstration content, designed to be replaced by cited historical records in a production catalogue.</p>
    </section></div>}
  </section>;
}
