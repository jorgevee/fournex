import { NextResponse } from "next/server";
import type { NextFetchEvent, NextRequest } from "next/server";
import { getSession } from "next-auth/react";

export async function middleware(req: NextRequest, event: NextFetchEvent) {
  // Call our authentication function to check the request
  const session = await getSession();
  if (!session) {
    // Respond with JSON indicating an error message
    return NextResponse.json(
      { success: false, message: "authentication failed" },
      { status: 401 },
    );
  }
}
// See "Matching Paths" below to learn more
export const config = {
  matcher: "/dashboard/:path",
};
