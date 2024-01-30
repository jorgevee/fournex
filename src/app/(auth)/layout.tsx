import { getServerAuthSession } from "~/server/auth";
import { redirect } from "next/navigation";

export const metadata = {
  title: "Fournex Auth",
  description: "...",
};
export default async function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getServerAuthSession();
  if (session) {
    redirect("/");
  }

  return <div className="container mx-auto">{children}</div>;
}
