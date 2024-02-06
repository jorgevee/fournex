import ContactForm from "./contactForm";

import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Contact Us',
  description: 'Contact Fournex for any questions',
};

const ContactPage = async () => {
  return (
    <div className="flex h-screen items-center justify-center bg-gray-100">
      <div className="w-full max-w-md rounded-lg bg-white p-8 shadow-md">
        <h1 className="mb-4 text-2xl font-bold text-blue-500">Contact Us</h1>
        <p className="mb-4 text-gray-600">Contact Fournex for any questions</p>
        <ContactForm />
      </div>
    </div>
  );
};

export default ContactPage;
