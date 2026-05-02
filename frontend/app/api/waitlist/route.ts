import { mkdir, writeFile } from "fs/promises";
import { NextRequest, NextResponse } from "next/server";
import path from "path";
import crypto from "crypto";

const WAITLIST_PATH = path.join(process.cwd(), "data", "waitlist.jsonl");

function isValidEmail(value: unknown): value is string {
  if (typeof value !== "string") {
    return false;
  }
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.trim());
}

export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const payload = body as Record<string, unknown>;
  const email = typeof payload.email === "string" ? payload.email.trim().toLowerCase() : "";

  if (!isValidEmail(email)) {
    return NextResponse.json({ error: "Enter a valid email address." }, { status: 400 });
  }

  const record = {
    id: crypto.randomUUID(),
    email,
    source: typeof payload.source === "string" ? payload.source : "waitlist",
    created_at: new Date().toISOString(),
  };

  await mkdir(path.dirname(WAITLIST_PATH), { recursive: true });
  await writeFile(WAITLIST_PATH, `${JSON.stringify(record)}\n`, {
    encoding: "utf-8",
    flag: "a",
  });

  return NextResponse.json({ ok: true, id: record.id });
}
