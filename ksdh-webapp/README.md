# ARC2 Web Application

This is the frontend for the **ARC2 (Agentic AI Research Review and Conference Classification)** project, built with the [T3 Stack](https://create.t3.gg/).

## 🚀 Overview

This Next.js application provides a user interface to interact with the ARC2 agent service. It allows users to upload research papers (in PDF format) or paste text content to:

-   **Evaluate Publishability**: Get a detailed analysis of the paper's strengths, weaknesses, and a publishability prediction.
-   **Get Conference Recommendations**: Receive a recommendation for the most suitable conference (from a predefined list: CVPR, EMNLP, KDD, NeurIPS, TMLR).
-   **Visualize Reasoning**: Explore the agent's Tree of Thoughts to understand its decision-making process.

## 🛠️ Tech Stack

-   [Next.js](https://nextjs.org)
-   [React](https://reactjs.org/)
-   [Tailwind CSS](https://tailwindcss.com)
-   [tRPC](https://trpc.io)
-   [Shadcn/ui](https://ui.shadcn.com/)

## ⚙️ Setup and Execution

### Prerequisites

-   Node.js and npm
-   The [ARC2 Agent Service](../agent/README.md) must be running.

### Running the Application

1.  Navigate to the `ksdh-webapp` directory:
    ```bash
    cd ksdh-webapp
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Run the development server:
    ```bash
    npm run dev
    ```
4.  Open [http://localhost:3000](http://localhost:3000) in your browser to view the application.

## 🌐 Deployment

This application is configured for easy deployment on platforms like Vercel or Netlify. Follow their respective deployment guides for Next.js applications.

-   [Vercel Deployment Guide](https://create.t3.gg/en/deployment/vercel)
-   [Netlify Deployment Guide](https://create.t3.gg/en/deployment/netlify)
