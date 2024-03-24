import { NavBar } from "./layoutClient";
import { getServerAuthSession } from "~/server/auth";

export default async function Header() {
  const session = await getServerAuthSession();
  return (
    <header className="bg-slate-900 p-4 text-white">
      <NavBar session={session ? true : false} />
    </header>
  );
}
