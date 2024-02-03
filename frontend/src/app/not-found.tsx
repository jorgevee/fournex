// app/not-found.tsx
import Link from "next/link";

const NotFoundPage = () => {
  return (
    <div className="container mx-auto flex flex-col items-center justify-center py-16">
      <h1 className="mb-6 text-4xl font-bold text-gray-700">
        Oops! That page doesn&apos;t exist.
      </h1>
      <p className="mb-8 max-w-xl text-lg text-gray-500">
        We couldn&apos;t find the page you were looking for. Perhaps you
        mistyped the URL or it was removed.
      </p>
      <Link href="/">
        <button className="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700">
          Go back to Home
        </button>
      </Link>
    </div>
  );
};

export default NotFoundPage;
