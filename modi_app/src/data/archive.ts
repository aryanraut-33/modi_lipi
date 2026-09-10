export type ArchiveItem = {
  title: string;
  period: string;
  type: "Fort" | "Battle" | "Document" | "Artefact";
  place: string;
  description: string;
  symbol: string;
};

export const archiveItems: ArchiveItem[] = [
  { title: "Raigad Fort", period: "1674 CE", type: "Fort", place: "Raigad, Maharashtra", symbol: "⌂", description: "The capital where Chhatrapati Shivaji Maharaj was formally crowned and Swarajya found its ceremonial centre." },
  { title: "Pratapgad", period: "1659 CE", type: "Fort", place: "Satara, Maharashtra", symbol: "▲", description: "A hill fort overlooking the Konkan route, remembered for the encounter that reshaped regional power." },
  { title: "The Coronation", period: "6 June 1674", type: "Document", place: "Raigad", symbol: "✦", description: "A moment of sovereign assertion: ritual, governance, and legitimacy converged at the summit of Raigad." },
  { title: "Battle of Sinhagad", period: "1670 CE", type: "Battle", place: "Pune, Maharashtra", symbol: "⚔", description: "An audacious night ascent that returned the fort to Maratha control and entered Maharashtra’s collective memory." },
  { title: "Rajmudra", period: "17th century", type: "Artefact", place: "Maratha court", symbol: "◉", description: "The royal seal, an enduring symbol of public welfare, self-rule, and responsibility." },
  { title: "Sindhudurg", period: "1664–67 CE", type: "Fort", place: "Malvan, Maharashtra", symbol: "≋", description: "A sea fort engineered to protect the Konkan coast and establish a strategic naval presence." }
];

export const milestones = [
  ["1630", "Birth at Shivneri", "The birth of Shivaji Maharaj at the hill fort of Shivneri."],
  ["1646", "Torna captured", "The first major fort acquired, marking the beginning of an expanding Swarajya."],
  ["1659", "Pratapgad", "A decisive encounter in the Western Ghats strengthened the Maratha position."],
  ["1670", "Sinhagad retaken", "A bold campaign returned the fort near Pune to Maratha control."],
  ["1674", "Coronation at Raigad", "Shivaji Maharaj was crowned Chhatrapati, establishing a sovereign kingdom."]
];
