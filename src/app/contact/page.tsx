import React from "react";

const ContactPage = () => {
  return (
    <div className="flex h-screen items-center justify-center bg-gray-100">
      <div className="w-full max-w-md rounded-lg bg-white p-8 shadow-md">
        <h1 className="mb-4 text-2xl font-bold text-blue-500">Contact Us</h1>
        <p className="mb-4 text-gray-600">Contact Fournex for any questions</p>

        <form>
          <div className="mb-4">
            <label htmlFor="name" className="block font-bold text-gray-700">
              Name
            </label>
            <input
              type="text"
              id="name"
              className="w-full rounded border border-gray-400 p-2"
            />
          </div>

          <div className="mb-4">
            <label htmlFor="email" className="block font-bold text-gray-700">
              Email
            </label>
            <input
              type="email"
              id="email"
              className="w-full rounded border border-gray-400 p-2"
              required={true}
            />
          </div>

          <div className="mb-4">
            <label htmlFor="message" className="block font-bold text-gray-700">
              Message
            </label>
            <textarea
              id="message"
              className="w-full rounded border border-gray-400 p-2"
              rows="4"
              required={true}
            ></textarea>
          </div>

          <button
            type="submit"
            className="w-full rounded bg-blue-500 px-4 py-2 text-white"
          >
            Submit
          </button>
        </form>
      </div>
    </div>
  );
};

export default ContactPage;
