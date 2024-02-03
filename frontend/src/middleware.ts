import { NextResponse } from "next/server";
import type { NextFetchEvent, NextRequest } from "next/server";

export function middleware(req: NextRequest, event: NextFetchEvent) {
  // Call our authentication function to check the request
  const token = req.cookies.get("next-auth.session-token");
  if (!token?.value) {
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
