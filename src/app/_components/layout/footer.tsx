import Link from "next/link";
export default function Footer() {
  return (
    <footer className="bg-slate-800 p-6 text-white">
      <div className="container mx-auto flex flex-wrap items-center">
        <div className="w-full text-center md:w-1/4 md:text-left">
          <h2 className="mb-4 text-lg font-bold">About Us</h2>
          <p>
            Fournex enables building stateful, multi-agent applications with
            LLMs. It coordinates multiple AI agents to interact in a cyclic
            workflow.
          </p>
        </div>
        <div className="w-full text-center md:w-1/4 md:text-left">
          <h2 className="mb-4 text-lg font-bold">Links</h2>
          <ul className="list-unstyled">
            <li>
              <Link href="/" className="text-gray-400 hover:text-white">
                Home
              </Link>
            </li>
            <li>
              <Link href="#" className="text-gray-400 hover:text-white">
                About
              </Link>
            </li>
            <li>
              <Link href="#" className="text-gray-400 hover:text-white">
                Services
              </Link>
            </li>
            <li>
              <Link href="/contact" className="text-gray-400 hover:text-white">
                Contact
              </Link>
            </li>
          </ul>
        </div>
        <div className="w-full text-center md:w-1/4 md:text-left">
          <h2 className="mb-4 text-lg font-bold">Contact Us</h2>
          <p>
            77 Geary St.
            <br />
            San Francisco, CA 94108
            <br />
            415-456-7890
            <br />
            info@fournex.com
          </p>
        </div>
        <div className="w-full text-center md:w-1/4 md:text-left">
          <h2 className="mb-4 text-lg font-bold">Follow Us</h2>
          <ul className="list-unstyled">
            <li>
              <Link href="#" className="text-gray-400 hover:text-white">
                <i className="fab fa-facebook-f"></i>
              </Link>
            </li>
            <li>
              <Link href="#" className="text-gray-400 hover:text-white">
                <i className="fab fa-twitter"></i>
              </Link>
            </li>
            <li>
              <Link href="#" className="text-gray-400 hover:text-white">
                <i className="fab fa-instagram"></i>
              </Link>
            </li>
            <li>
              <Link href="#" className="text-gray-400 hover:text-white">
                <i className="fab fa-linkedin-in"></i>
              </Link>
            </li>
          </ul>
        </div>
      </div>
      <div className="container mx-auto py-4 text-center">
        <p>&copy; 2024 Fournex. All Rights Reserved. </p>
        <a
          className="ext-blue-500 hover:text-blue-700"
          href="https://github.com/jorgevee"
        >
          Developed by Jorge Villegas
        </a>
      </div>
    </footer>
  );
}
