# AutoML RL Retraining Controller

A comprehensive web application for automated machine learning with reinforcement learning-based retraining control.

## Features

- **Dataset Management**: Upload and manage CSV datasets with drag-and-drop support
- **Baseline Training**: Train classifier models with multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- **Optuna Hyperparameter Tuning**: Automated hyperparameter optimization with visualization
- **RL Controller**: Reinforcement learning-based controller for automated retraining decisions
- **Episode Simulation**: Run multiple episodes to evaluate controller performance
- **Model Registry**: Track all trained models with performance metrics
- **Comprehensive Logging**: Filterable activity logs for all system components

## Getting Started

1. **Upload a Dataset**: Navigate to the Datasets page and drag & drop a CSV file
2. **Train a Baseline Model**: Go to the Training page and train an initial model
3. **Run Controller**: Use the Controller page to execute step-by-step decisions or simulate episodes
4. **Monitor Performance**: View the Dashboard for KPIs and performance metrics

## RL Model Integration

The application is designed to work with a trained PPO (Proximal Policy Optimization) RL model. Currently, it uses a heuristic placeholder that makes decisions based on:
- Performance drift threshold
- Steps since last retrain
- Current model accuracy

To integrate your uploaded PPO model (`ppo_automan_v1_1763487706_1763489591506.zip`), you would need to:
1. Set up a Python bridge service that can load the stable-baselines3 model
2. Expose an API endpoint that accepts state observations and returns actions
3. Update the controller routes to call this service instead of the heuristic

## Technology Stack

- **Frontend**: React, TypeScript, Tailwind CSS, Shadcn UI, Recharts
- **Backend**: Express.js, TypeScript
- **Storage**: In-memory storage (can be upgraded to PostgreSQL)
- **ML Integration**: Designed for Python ML stack (scikit-learn, Optuna, stable-baselines3)

## Architecture

The application follows a schema-first design:
- **Shared Types**: TypeScript interfaces ensure type safety between frontend and backend
- **RESTful API**: Clean separation between data layer and presentation
- **Reactive UI**: Real-time updates using React Query with automatic cache invalidation
- **Professional Design**: ML dashboard aesthetic with Inter and JetBrains Mono fonts
