"use client";
import { useState } from "react";

export default function ContactForm() {
  const [error, setError] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError("There is currently no email API yet");
  };
  return (
    <form onSubmit={handleSubmit}>
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
          required={true}
        ></textarea>
      </div>

      <button
        type="submit"
        className="w-full rounded bg-blue-500 px-4 py-2 text-white"
      >
        Submit
      </button>
      {error && <div className="text-red-500">{error}</div>}
    </form>
  );
}
