"use client";

interface ImageModalProps {
  src: string;
  title: string;
  onClose: () => void;
}

export function ImageModal({ src, title, onClose }: ImageModalProps) {
  return (
    <div
      className="image-modal"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onClick={onClose}
    >
      <figure onClick={(e) => e.stopPropagation()}>
        <button onClick={onClose} aria-label="Close enlarged image">
          ×
        </button>
        <img src={src} alt={title} />
        <figcaption>{title}</figcaption>
      </figure>
    </div>
  );
}
