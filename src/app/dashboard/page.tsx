import { getServerAuthSession } from "~/server/auth";

export default async function Dashboard() {
  const session = await getServerAuthSession();

  return (
    <div className="flex h-screen flex-col items-center justify-center">
      <h1 className="text-3xl font-bold">Dashboard</h1>
      <p className="text-gray-600">Hello welcome {session?.user?.email}</p>
    </div>
  );
}
