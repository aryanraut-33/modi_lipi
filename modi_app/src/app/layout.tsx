import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Modi ShatVidhi | Transliteration Workspace",
  description: "A web workspace for the Modi ShatVidhi manuscript preprocessing and segmentation pipeline."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
