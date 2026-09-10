"use client";
import { useState } from "react";
import { milestones } from "../data/archive";

export function Timeline() {
  const [active, setActive] = useState(4);
  const item = milestones[active];
  return <section className="timeline section" id="timeline" aria-labelledby="timeline-title">
    <div className="eyebrow">A sovereign journey</div><div className="section-heading"><div><h2 id="timeline-title">A life in the making of Swarajya</h2><p>Select a year to read a defining moment in the Maratha chronicle.</p></div><span className="script">स्वराज्य</span></div>
    <div className="timeline-layout"><div className="timeline-rail" role="tablist" aria-label="Historical timeline">{milestones.map(([year, title], index) => <button key={year} role="tab" aria-selected={active === index} onClick={() => setActive(index)}><strong>{year}</strong><span>{title}</span></button>)}</div>
      <article className="timeline-detail"><p className="detail-year">{item[0]} <span>CE</span></p><h3>{item[1]}</h3><p>{item[2]}</p><a href="#explore">View related records <span>→</span></a></article></div>
  </section>;
}
