export const metadata = {
  title: "Fournex Auth",
  description: "...",
};
export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <div className="container mx-auto">{children}</div>;
}
