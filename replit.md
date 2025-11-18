# AutoML RL Retraining Controller

## Overview

This is a web-based AutoML platform that combines traditional machine learning workflows with reinforcement learning-based retraining control. The application enables users to upload datasets, train baseline classifier models, optimize hyperparameters using Optuna, and automate retraining decisions through an RL controller. The system is designed as a technical ML operations tool with a focus on data density, clarity, and workflow efficiency.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack:**
- React 18 with TypeScript for type-safe component development
- Wouter for lightweight client-side routing
- TanStack Query (React Query) for server state management with automatic cache invalidation
- Recharts for data visualization (performance charts, Optuna trial visualization)

**UI Framework:**
- Shadcn UI component library built on Radix UI primitives
- Tailwind CSS for utility-first styling with custom design tokens
- Design system based on modern ML dashboard aesthetics (inspired by Weights & Biases, TensorBoard, Linear)
- Typography: Inter for UI elements, JetBrains Mono for code/logs/model IDs

**State Management:**
- React Query handles all server state with configurable cache settings (staleTime: Infinity, no automatic refetching)
- Local component state for UI interactions (selected dataset, algorithm choices, filter states)
- No global state management library - server state is the source of truth

**Layout System:**
- Fixed sidebar navigation (16rem width) with collapsible state
- Main content area with max-width constraint (max-w-7xl)
- 12-column grid system for dashboard metrics and charts
- Responsive breakpoints managed through Tailwind

### Backend Architecture

**Technology Stack:**
- Express.js with TypeScript for REST API server
- Node.js runtime with ES modules
- HTTP server for both API routes and static file serving

**API Design:**
- RESTful endpoints organized by domain (datasets, models, episodes, controller, logs, optuna)
- JSON request/response format
- Multipart form-data support for file uploads (via multer)
- Request/response logging middleware with duration tracking

**Data Layer:**
- Storage abstraction interface (IStorage) allows swapping implementations
- Currently uses in-memory storage with synchronous Map-based data structures
- Schema-first design with shared TypeScript types between frontend and backend
- Drizzle ORM configured for PostgreSQL migration path (schema defined, database not yet provisioned)

**Business Logic:**
- Baseline model training with multiple algorithm support (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- Heuristic-based RL controller placeholder (designed to be replaced with actual PPO model integration)
- Episode simulation for evaluating controller performance over multiple timesteps
- Activity logging system with component-level categorization

### Data Storage

**Current Implementation:**
- In-memory storage using JavaScript Map objects
- Data structures: Dataset, Model, Episode, ControllerStep, Log, OptunaTrial
- UUID-based primary keys
- No persistence between server restarts

**Designed Migration Path:**
- Drizzle ORM schema defined for PostgreSQL in `shared/schema.ts`
- Tables use `gen_random_uuid()` for auto-generated IDs
- JSONB columns for flexible metadata storage (hyperparameters, metrics, feature transformations)
- Migration configuration ready in `drizzle.config.ts`
- Connection pooling ready via @neondatabase/serverless

**Schema Design:**
- Datasets: CSV metadata with column information
- Models: Training runs with algorithm, hyperparameters, and performance metrics
- Episodes: RL controller simulation runs tracking rewards and retrain counts
- ControllerSteps: Individual decision points within episodes
- OptunaTrials: Hyperparameter optimization trial history
- Logs: System-wide activity logging with level, component, and message fields

### External Dependencies

**UI Component Libraries:**
- Radix UI primitives (@radix-ui/*): Accessible, unstyled component foundation
- Recharts: Chart rendering for performance metrics and Optuna visualizations
- cmdk: Command palette interface support
- class-variance-authority: Type-safe component variant management
- Tailwind CSS with PostCSS: Utility-first styling

**Development Tooling:**
- Vite: Frontend build tool and dev server with HMR
- esbuild: Backend bundling for production
- TypeScript compiler: Type checking across client/server/shared code
- Drizzle Kit: Database schema management and migrations

**Planned ML Integration:**
- Python bridge service (not yet implemented) for:
  - scikit-learn model training
  - Optuna hyperparameter optimization
  - stable-baselines3 PPO model inference
- Design expects HTTP API between Node.js backend and Python ML service

**Google Fonts CDN:**
- Inter font family (weights: 400, 500, 600, 700)
- JetBrains Mono (weights: 400, 500)
- Loaded via HTML link tags in client/index.html

**Session Management:**
- connect-pg-simple: PostgreSQL session store (configured but database not provisioned)
- express-session support prepared for future authentication

**Database:**
- PostgreSQL (schema defined, not yet provisioned)
- @neondatabase/serverless: Serverless Postgres driver
- Connection expected via DATABASE_URL environment variable