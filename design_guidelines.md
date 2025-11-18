# AutoML RL Retraining Controller - Design Guidelines

## Design Approach
**System-Based Approach** using modern data dashboard principles inspired by Weights & Biases, TensorBoard, and Linear. This is a technical ML operations tool where clarity, data density, and workflow efficiency are paramount.

## Layout System

**Spacing Primitives**: Use Tailwind units of `2, 4, 6, 8, 12, 16, 20` for consistent rhythm
- Compact spacing: `p-2, gap-4` for dense data displays
- Standard spacing: `p-6, gap-8` for main content areas
- Generous spacing: `p-12, p-16` for section separation

**Grid Structure**: 
- Sidebar navigation (280px fixed): Model registry, logs, settings
- Main content area: Flexible width with max-w-7xl container
- Dashboard uses 12-column grid for metric cards and charts
- Two-column layouts for upload/preview, training/results sections

## Typography

**Font Families** (via Google Fonts CDN):
- Primary: `Inter` - for UI, data labels, metrics (weights: 400, 500, 600, 700)
- Monospace: `JetBrains Mono` - for code, logs, model IDs (weights: 400, 500)

**Hierarchy**:
- Page titles: text-3xl font-bold
- Section headers: text-xl font-semibold
- Subsections: text-lg font-medium
- Body/labels: text-sm font-medium
- Metadata/timestamps: text-xs
- Logs/code: text-sm font-mono

## Component Library

### Navigation
- **Fixed Sidebar**: Icon + label navigation items, collapsible sections for model versions, active state indicator
- **Top Bar**: Breadcrumb navigation, global actions (upload, new training), user/settings menu

### Core UI Elements

**Cards**: Elevated containers with subtle borders for grouping related information
- Metric cards: Large number display + label + trend indicator (↑↓)
- Section cards: Header with actions + content area + optional footer
- Status cards: Icon + status label + timestamp + details link

**Data Tables**: 
- Sortable columns with icons
- Row hover states
- Action buttons per row (view, retrain, delete)
- Pagination controls
- Inline status badges

**Forms & Inputs**:
- File upload: Drag-and-drop zone with file preview
- Parameter inputs: Labels above inputs, helper text below
- Sliders: For threshold adjustments with current value display
- Toggles: Enable/disable features with immediate visual feedback

**Buttons**:
- Primary: Solid fill for main actions (Train Model, Run Episode)
- Secondary: Outlined for secondary actions (Cancel, Reset)
- Icon buttons: For table actions and toolbar controls
- Button groups: For related actions (Step/Run/Pause controller)

### Data Visualization Components

**Metrics Dashboard**:
- 4-column grid (responsive to 2-col, 1-col) for KPI cards
- Large metric values with trend indicators and sparklines
- Performance over time: Line charts showing accuracy/loss curves
- Optuna optimization: Parallel coordinates plot, hyperparameter importance

**Controller Execution View**:
- State display: Current environment state as structured JSON viewer
- Action selection: Visual representation of RL decision (retrain/wait/tune)
- Step controls: Previous/Next/Play/Pause buttons
- Timeline: Horizontal steps with state transitions

**Episode Simulation**:
- Configuration panel: Episode count, max steps, parameters
- Progress indicator: Current episode/step with ETA
- Live metrics: Updating reward chart, cumulative performance
- Results summary: Table of episode outcomes

**Logs Viewer**:
- Filterable list: By level (info/warning/error), timestamp, component
- Monospace formatting with syntax highlighting for structured logs
- Auto-scroll toggle, search functionality
- Expandable log entries for detailed stack traces

### Model Registry

**Registry List**: Table view with columns:
- Model ID (monospace), timestamp, metrics (accuracy, F1), status badge, actions
- Version comparison: Side-by-side metric comparison
- Download/load model controls

**Detail View**: 
- Two-column: Metadata (left) + Performance charts (right)
- Hyperparameters table, training configuration, feature transformations applied
- Lineage graph: Parent models and retraining history

### Special Components

**Dataset Preview**: 
- CSV table with fixed header, scrollable body
- Column statistics sidebar (min/max/mean for numeric, distribution for categorical)
- Sample size indicator

**Feature Transformation Pipeline**:
- Vertical flow diagram showing transformation steps
- Add/remove transformation controls
- Preview toggle to see transformed data

## Page Layouts

**Main Dashboard**: 
- Top: 4 KPI metric cards (Current Accuracy, Models Trained, Retraining Events, Avg Reward)
- Middle: 2-column (Performance Chart | Recent Episodes Table)
- Bottom: Recent Logs preview with "View All" link

**Upload & Train**:
- Left column: Upload zone + dataset preview
- Right column: Training configuration form + baseline metrics after training

**Controller Execution**:
- Full-width state viewer at top
- Center: Large action display with decision reasoning
- Bottom: Step controls + episode timeline

**Optuna Tuner**:
- Configuration sidebar (left 30%)
- Main area: Optimization plots + best parameters table + trials history

**Model Registry**:
- Full-width table with filters/search header
- Modal overlay for detailed model view

## Accessibility & Interaction

- All interactive elements meet WCAG AA contrast standards
- Keyboard navigation with visible focus states
- Form validation with inline error messages
- Loading states: Skeleton screens for tables, spinners for async operations
- Empty states: Helpful messages with action prompts ("No datasets uploaded yet. Upload your first dataset to begin.")

## Icons
Use **Heroicons** (outline style) via CDN for consistency:
- Navigation: home, chart-bar, cog, folder, document-text
- Actions: upload, play, pause, refresh, trash, download
- Status: check-circle, exclamation-triangle, information-circle
- Data: table-cells, beaker, cpu-chip

## Visual Principles
- **Information Density**: Maximize useful data per screen without clutter
- **Hierarchy Through Contrast**: Use size, weight, and spacing—not just visual treatment
- **Functional Aesthetics**: Every visual element serves the workflow
- **Responsive Behavior**: Tables scroll horizontally on mobile, cards stack vertically
- **Consistent Patterns**: Reuse card layouts, button styles, and spacing across all views