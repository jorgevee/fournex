# Fournex: Building Autonomous Agents with LangGraph

Fournex is a full-stack application that utilizes large language models (LLMs) and LangGraph to build autonomous agents. LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.

## Key Features

- Add cycles to your LLM application
- NOT a DAG framework
- Agent-like behaviors with LLMs

## Tech Stack

- Next.js with app router (client)
- Tailwind (CSS)
- FastAPI (API)
- LangChain (agent building with LLMs)
- Postgres (database)
- tRPC

## Getting Started

1. Clone the repository

   ```bash
   git clone https://github.com/your-username/fournex.git
   ```

2. Install dependencies for the client

   ```bash
   cd fournex/frontend
   npm install
   ```

3. Install dependencies for the server

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Set up the database

   - Create a new Postgres database and user
   - Update the `DATABASE_URL` environment variable in `.env` file

   We recommend using Supabase for managing your Postgres database due to its ease of use and powerful features.

5. Run migrations

   ```bash
   python manage.py migrate
   ```

6. Start the development server

- For the client
  ```
  npm run dev
  ```
- For the server
  ```
  python manage.py runserver
  ```

### Prerequisites

- Node.js (>= 14.x)
- Python (>= 3.8)
- Postgres

### Installation

1. Clone the repository

## Usage

- To use LangGraph, you can create a new `Graph` instance and add `Chain` actors to it.
- Each `Chain` actor represents a step in the computation and can be connected to other actors to form a cycle.
- You can then execute the graph and retrieve the results.
  -This project uses LangServe to expose LangChain runnables and chains as a REST API, powered by the fantastic FastAPI framework. Here's an overview of how it works:

## LangServe & FastAPI Integration

- LangServe seamlessly integrates with FastAPI, allowing you to deploy your LangChain logic as endpoints with ease. This means:

  Automatic validation: Input and output schemas are automatically inferred from your LangChain objects, ensuring correct data format and providing rich error messages during API calls.
  Comprehensive API docs: Enjoy comprehensive API documentation generated with JSONSchema and Swagger, making it easy for developers to understand and utilize your endpoints.
  Flexible access: Choose from various endpoints like /invoke/, /batch/, and /stream/ to handle different request types and concurrent calls efficiently.

## Contributing

We welcome contributions to Fournex! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
