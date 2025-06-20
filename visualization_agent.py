import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import together
import json
import re

class VisualizationAgent:
    """
    Intelligent visualization agent that creates appropriate charts based on data and queries
    """
    
    def __init__(self, together_api_key: str):
        self.together_client = together.Together(api_key=together_api_key)
        self.model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        
        # Visualization keywords
        self.viz_keywords = {
            'chart', 'plot', 'graph', 'visualization', 'visualize', 'show', 'display',
            'histogram', 'scatter', 'line', 'bar', 'pie', 'box', 'heatmap', 'distribution'
        }
        
        # Chart type mapping
        self.chart_types = {
            'scatter': self._create_scatter_plot,
            'line': self._create_line_plot,
            'bar': self._create_bar_plot,
            'histogram': self._create_histogram,
            'box': self._create_box_plot,
            'pie': self._create_pie_chart,
            'heatmap': self._create_heatmap,
            'correlation': self._create_correlation_matrix
        }
    
    def should_create_visualization(self, query: str) -> bool:
        """Determine if a visualization should be created based on the query"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.viz_keywords)
    
    def create_visualization(self, query: str, processed_data: Dict[str, Any], context: str = "") -> Optional[go.Figure]:
        """Create appropriate visualization based on query and data"""
        try:
            # Get structured data
            structured_data = self._get_structured_data(processed_data)
            
            if not structured_data:
                return None
            
            # Determine visualization type and parameters
            viz_plan = self._plan_visualization(query, structured_data, context)
            
            if not viz_plan:
                return None
            
            # Create the visualization
            return self._execute_visualization_plan(viz_plan, structured_data)
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None
    
    def _get_structured_data(self, processed_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Extract structured data (DataFrames) from processed data"""
        structured_data = {}
        
        for file_name, data_info in processed_data.items():
            if data_info.get('type') == 'structured':
                structured_data[file_name] = data_info['data']
        
        return structured_data
    
    def _extract_last_valid_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Safely extract the last valid JSON object from the LLM response."""
        import json
        import re

        # Try to find JSON blocks in ```json ... ``` format
        json_blocks = re.findall(r'```json(.*?)```', response_text, re.DOTALL)
        for block in reversed(json_blocks):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue

        # Fallback: try to find brace-based JSON
        brace_blocks = re.findall(r'\{.*\}', response_text, re.DOTALL)
        for block in reversed(brace_blocks):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue

        return None

    def _plan_visualization(self, query: str, structured_data: Dict[str, pd.DataFrame], context: str) -> Optional[Dict[str, Any]]:
        """Plan the visualization using AI"""
        try:
            # Prepare data summary for the AI
            data_summary = self._prepare_data_summary(structured_data)
            
            prompt = f"""You are an expert data visualization specialist. Based on the user's query and available data, create a visualization plan.
Your task is to generate a JSON plan for a chart based on the user's query and available structured datasets.
User Query: {query}

Available Data:
{data_summary}

Context: {context}

Instructions:
1. Determine the most appropriate chart type (scatter, line, bar, histogram, box, pie, heatmap, correlation)
2. Select the relevant dataset and columns
3. Provide specific parameters for the visualization

Respond with a JSON object containing:
{{
  "chart_type": "type_of_chart",
  "dataset": "dataset_name",
  "x_column": "column_name",
  "y_column": "column_name",
  "color_column": "column_name or null",
  "title": "Chart Title",
  "reasoning": "Why this visualization is appropriate",
  "filter": {{
    "column": "column_to_filter",
    "operation": "top_n | bottom_n | greater_than | less_than | equals | not_equals | between | contains | not_contains",
    "value": <number, string, or list depending on operation>
  }},
  "x_axis_format": {{
    "tick_format": "~s | .2f | $,.0f | custom",
    "unit": "freeform based on context (e.g., raw numbers, millions, billions, tonnes, litres, etc)",
    "label": "Human-readable x-axis label"
  }},
  "y_axis_format": {{
    "tick_format": "~s | .2f | $,.0f | custom",
    "unit": "freeform based on context (e.g., millions of dollars,, billions, tonnes, litres, etc)",
    "label": "Human-readable y-axis label"
  }}
}}

If no visualization is appropriate, respond with {{"chart_type": "none"}}.

Be sure to:
- Choose axis format based on the actual numeric range (e.g., 500M → millions)
- Use tick_format to keep it readable (e.g., '~s' for SI units, '$,.0f' for dollars)
- Label axes in a user-friendly way

Response:"""
            print("[DEBUG] Prompt sent to Together LLM:\n", prompt)
            # Call the LLM to get the visualization plan
            response = self.together_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
                stop=None,
            )
            response_text = response.choices[0].message.content.strip()
            print("[DEBUG] Raw LLM response for visualization plan:\n", response_text)
            
            # Extract JSON from response
            viz_plan = self._extract_last_valid_json(response_text)
            try:
                if viz_plan:
                    if viz_plan.get("chart_type") != "none":
                        return viz_plan
                    else:
                        print("[INFO] Visualization plan returned 'none'.")
            except Exception as e:
                print(f"[ERROR] Could not extract a valid JSON visualization plan: {str(e)}")

            return None
            
        except Exception as e:
            print(f"Error planning visualization: {str(e)}")
            return None
    
    def _prepare_data_summary(self, structured_data: Dict[str, pd.DataFrame]) -> str:
        """Prepare a summary of available data for the AI"""
        summary = ""
        
        for dataset_name, df in structured_data.items():
            summary += f"\nDataset: {dataset_name}\n"
            summary += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
            summary += "Columns:\n"
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                unique_count = df[col].nunique()
                
                if df[col].dtype in ['int64', 'float64']:
                    summary += f"  - {col} ({col_type}): Range {df[col].min():.2f} to {df[col].max():.2f}\n"
                else:
                    summary += f"  - {col} ({col_type}): {unique_count} unique values\n"
                    if unique_count <= 10:
                        sample_values = df[col].dropna().unique()[:10]
                        summary += f"    Values: {', '.join(map(str, sample_values))}\n"
        
        return summary
    
    def _debug_data_before_filtering(self, df: pd.DataFrame, filter_cfg: Dict[str, Any]) -> None:
        """Debug data before applying filters"""
        print(f"\n[DEBUG] === DATA BEFORE FILTERING ===")
        print(f"Dataset shape: {df.shape}")
        
        if filter_cfg:
            col = filter_cfg.get("column")
            if col and col in df.columns:
                print(f"Filter column '{col}' statistics:")
                if df[col].dtype in ['int64', 'float64']:
                    print(f"  Min: {df[col].min()}")
                    print(f"  Max: {df[col].max()}")
                    print(f"  Mean: {df[col].mean():.2f}")
                    print(f"  Unique values: {df[col].nunique()}")
                    print(f"  Top 10 values:\n{df[col].nlargest(10).tolist()}")
                else:
                    print(f"  Unique values: {df[col].nunique()}")
                    print(f"  Value counts:\n{df[col].value_counts().head(10)}")
        
        print(f"[DEBUG] === END PRE-FILTER DEBUG ===\n")
    
    def _apply_filters(self, df: pd.DataFrame, filter_cfg: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the dataframe with enhanced debugging"""
        if not filter_cfg:
            return df
        
        # Debug data before filtering
        self._debug_data_before_filtering(df, filter_cfg)
        
        col = filter_cfg.get("column")
        op = filter_cfg.get("operation")
        val = filter_cfg.get("value")

        print(f"[DEBUG] Applying filter: column='{col}', operation='{op}', value='{val}'")

        if not isinstance(col, str) or col not in df.columns:
            print(f"[WARN] Invalid column or column '{col}' not in dataset.")
            return df

        try:
            original_size = len(df)
            
            if op == "top_n":
                if val is not None:
                    # Sort by the column in descending order and take top n
                    filtered_df = df.nlargest(int(val), col)
                    print(f"[DEBUG] Applied top_{val} filter on '{col}', got {len(filtered_df)} rows")
                    print(f"[DEBUG] Filtered data preview:\n{filtered_df[[col] + [c for c in df.columns if c != col][:2]].head()}")
                    return filtered_df
                else:
                    print(f"[WARN] 'top_n' operation requires a non-None value for 'val'.")
                    return df
            elif op == "bottom_n":
                if val is not None:
                    filtered_df = df.nsmallest(int(val), col)
                    print(f"[DEBUG] Applied bottom_{val} filter on '{col}', got {len(filtered_df)} rows")
                    return filtered_df
                else:
                    print(f"[WARN] 'bottom_n' operation requires a non-None value for 'val'.")
                    return df
            elif op == "greater_than":
                filtered_df = df[df[col] > val]
                print(f"[DEBUG] Applied greater_than filter, {original_size} -> {len(filtered_df)} rows")
                return filtered_df
            elif op == "less_than":
                filtered_df = df[df[col] < val]
                print(f"[DEBUG] Applied less_than filter, {original_size} -> {len(filtered_df)} rows")
                return filtered_df
            elif op == "equals":
                filtered_df = df[df[col] == val]
                print(f"[DEBUG] Applied equals filter, {original_size} -> {len(filtered_df)} rows")
                return filtered_df
            elif op == "not_equals":
                filtered_df = df[df[col] != val]
                print(f"[DEBUG] Applied not_equals filter, {original_size} -> {len(filtered_df)} rows")
                return filtered_df
            elif op == "between":
                if isinstance(val, list) and len(val) == 2:
                    filtered_df = df[df[col].between(val[0], val[1])]
                    print(f"[DEBUG] Applied between filter, {original_size} -> {len(filtered_df)} rows")
                    return filtered_df
            elif op == "contains":
                filtered_df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]
                print(f"[DEBUG] Applied contains filter, {original_size} -> {len(filtered_df)} rows")
                return filtered_df
            elif op == "not_contains":
                filtered_df = df[~df[col].astype(str).str.contains(str(val), case=False, na=False)]
                print(f"[DEBUG] Applied not_contains filter, {original_size} -> {len(filtered_df)} rows")
                return filtered_df
        except Exception as e:
            print(f"[ERROR] Failed applying filter {filter_cfg}: {e}")
            return df

        return df

    def _clean_and_validate_data(self, df: pd.DataFrame, x_col: str, y_col: str = None) -> pd.DataFrame:
        """Clean and validate data before visualization"""
        print(f"\n[DEBUG] === DATA CLEANING ===")
        print(f"Original data shape: {df.shape}")
        
        df_clean = df.copy()
        
        # Remove rows where x_col is null
        if x_col and x_col in df_clean.columns:
            before_count = len(df_clean)
            df_clean = df_clean.dropna(subset=[x_col])
            print(f"After removing null {x_col}: {before_count} -> {len(df_clean)} rows")
        
        # Remove rows where y_col is null if it exists
        if y_col and y_col in df_clean.columns:
            before_count = len(df_clean)
            df_clean = df_clean.dropna(subset=[y_col])
            print(f"After removing null {y_col}: {before_count} -> {len(df_clean)} rows")
            
            # Convert to numeric if possible
            original_dtype = df_clean[y_col].dtype
            df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
            print(f"Converted {y_col} from {original_dtype} to {df_clean[y_col].dtype}")
            
            # Remove rows where conversion failed
            before_count = len(df_clean)
            df_clean = df_clean.dropna(subset=[y_col])
            print(f"After removing non-numeric {y_col}: {before_count} -> {len(df_clean)} rows")
            
            # Show final data stats
            print(f"Final {y_col} stats: min={df_clean[y_col].min()}, max={df_clean[y_col].max()}, unique={df_clean[y_col].nunique()}")
        
        print(f"[DEBUG] === END DATA CLEANING ===\n")
        return df_clean

    def _execute_visualization_plan(self, viz_plan: Dict[str, Any], structured_data: Dict[str, pd.DataFrame]) -> Optional[go.Figure]:
        """Execute the visualization plan"""
        try:
            chart_type = viz_plan.get('chart_type')
            dataset_name = viz_plan.get('dataset')
            
            if not isinstance(chart_type, str) or dataset_name not in structured_data:
                return None
            
            print(f"\n[DEBUG] === EXECUTING VISUALIZATION PLAN ===")
            print(f"Chart type: {chart_type}")
            print(f"Dataset: {dataset_name}")
            print(f"Viz plan: {viz_plan}")
            
            df = structured_data[dataset_name].copy()
            print(f"Original dataset shape: {df.shape}")

            # Apply Filters
            if viz_plan.get("filter"):
                df = self._apply_filters(df, viz_plan["filter"])
                print(f"After filtering: {df.shape}")

            # Get the appropriate chart creation function
            chart_func = self.chart_types.get(chart_type)
            if not chart_func:
                print(f"[ERROR] No chart function found for type: {chart_type}")
                return None
            
            # Create the chart
            result = chart_func(df, viz_plan)
            print(f"[DEBUG] === END VISUALIZATION EXECUTION ===\n")
            return result
            
        except Exception as e:
            print(f"Error executing visualization plan: {str(e)}")
            return None
    
    def _create_scatter_plot(self, df: pd.DataFrame, viz_plan: Dict[str, Any]) -> Optional[go.Figure]:
        """Create scatter plot"""
        x_col = viz_plan.get('x_column')
        y_col = viz_plan.get('y_column')
        color_col = viz_plan.get('color_column')
        title = viz_plan.get('title', 'Scatter Plot')
        
        # Clean and validate data
        df = self._clean_and_validate_data(df, x_col, y_col)
        
        if df.empty:
            print("[WARN] No valid data for scatter plot after cleaning")
            return None

        print("[DEBUG] Data used for scatter plot:")
        print(df[[x_col, y_col]].head())
        
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_col if color_col and color_col in df.columns else None,
            title=title,
            template="plotly_white"
        )
        
        # Handle axis formatting
        x_fmt = viz_plan.get("x_axis_format", {})
        y_fmt = viz_plan.get("y_axis_format", {})

        fig.update_layout(
            xaxis=dict(
                title=x_fmt.get("label", x_col),
                tickformat=x_fmt.get("tick_format", None)
            ),
            yaxis=dict(
                title=y_fmt.get("label", y_col),
                tickformat=y_fmt.get("tick_format", "~s")
            )
        )

        return fig
    
    def _create_line_plot(self, df: pd.DataFrame, viz_plan: Dict[str, Any]) -> Optional[go.Figure]:
        """Create line plot"""
        x_col = viz_plan.get('x_column')
        y_col = viz_plan.get('y_column')
        title = viz_plan.get('title', 'Line Plot')
        
        # Clean and validate data
        df = self._clean_and_validate_data(df, x_col, y_col)
        
        if df.empty:
            print("[WARN] No valid data for line plot after cleaning")
            return None
            
        # Sort by x column for proper line plotting
        df = df.sort_values(by=x_col)

        print("[DEBUG] Data used for line plot:")
        print(df[[x_col, y_col]].head())
        
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col,
            title=title,
            template="plotly_white"
        )
        
        # Handle axis formatting
        x_fmt = viz_plan.get("x_axis_format", {})
        y_fmt = viz_plan.get("y_axis_format", {})

        fig.update_layout(
            xaxis=dict(
                title=x_fmt.get("label", x_col),
                tickformat=x_fmt.get("tick_format", None)
            ),
            yaxis=dict(
                title=y_fmt.get("label", y_col),
                tickformat=y_fmt.get("tick_format", "~s")
            )
        )

        return fig
    
    def _create_bar_plot(self, df: pd.DataFrame, viz_plan: Dict[str, Any]) -> Optional[go.Figure]:
        """Create bar plot with enhanced debugging and proper scaling"""
        x_col = viz_plan.get('x_column')
        y_col = viz_plan.get('y_column')
        title = viz_plan.get('title', 'Bar Chart')

        print(f"\n[DEBUG] === BAR PLOT CREATION ===")
        print(f"x_column: {x_col}, y_column: {y_col}")

        if not x_col or x_col not in df.columns:
            # Auto-select categorical column
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                x_col = categorical_cols[0]
                print(f"[DEBUG] Auto-selected x_column: {x_col}")
            else:
                print("[ERROR] No suitable x_column found")
                return None

        if not y_col or y_col not in df.columns:
            # Use count if no y column specified
            print("[DEBUG] No y_column specified, using value counts")
            value_counts = df[x_col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=title,
                template="plotly_white"
            )
            fig.update_xaxes(title=x_col)
            fig.update_yaxes(title="Count")
            return fig
        
        # Clean and validate data
        df = self._clean_and_validate_data(df, x_col, y_col)
        
        if df.empty:
            print("[WARN] No valid data for bar plot after cleaning")
            return None

        print("[DEBUG] Data used for bar plot:")
        print(df[[x_col, y_col]])
        print(f"[DEBUG] Y-column '{y_col}' values: min={df[y_col].min()}, max={df[y_col].max()}, unique_count={df[y_col].nunique()}")
        
        # Check if all values are the same
        if df[y_col].nunique() == 1:
            print(f"[WARNING] All values in '{y_col}' are identical ({df[y_col].iloc[0]})")
            print("[WARNING] This will result in bars of equal height")
            print("[SUGGESTION] Check your data source - this might indicate a data processing issue")
            
            # Show raw data to help debug
            print(f"[DEBUG] Raw data sample:")
            print(df.head(10))

        # Create the bar plot
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title,
            template="plotly_white"
        )

        # Apply axis formatting
        x_fmt = viz_plan.get("x_axis_format", {})
        y_fmt = viz_plan.get("y_axis_format", {})

        # Calculate proper y-axis range with some padding
        y_min = df[y_col].min()
        y_max = df[y_col].max()
        
        if y_min == y_max:
            # When all values are the same, create a range around that value
            padding = abs(y_max) * 0.1 if y_max != 0 else 1000000  # 10% padding or 1M if value is 0
            y_axis_range = [y_max - padding, y_max + padding]
            print(f"[DEBUG] All values identical, using range: {y_axis_range}")
        else:
            y_range = y_max - y_min
            padding = y_range * 0.1 if y_range > 0 else y_max * 0.1
            y_axis_range = [max(0, y_min - padding), y_max + padding]
            print(f"[DEBUG] Normal range calculation: {y_axis_range}")
        
        fig.update_layout(
            xaxis=dict(
                title=x_fmt.get("label", x_col),
                tickformat=x_fmt.get("tick_format", None)
            ),
            yaxis=dict(
                title=y_fmt.get("label", y_col if y_col else "Count"),
                tickformat=y_fmt.get("tick_format", "~s"),
                range=y_axis_range
            )
        )
        
        print(f"[DEBUG] === END BAR PLOT CREATION ===\n")
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, viz_plan: Dict[str, Any]) -> Optional[go.Figure]:
        """Create histogram"""
        x_col = viz_plan.get('x_column')
        title = viz_plan.get('title', 'Histogram')
        
        # Clean and validate data
        df = self._clean_and_validate_data(df, x_col)
        
        if df.empty:
            print("[WARN] No valid data for histogram after cleaning")
            return None

        print("[DEBUG] Data used for histogram:")
        print(df[x_col].describe())
        
        fig = px.histogram(
            df,
            x=x_col,
            title=title,
            template="plotly_white"
        )
        
        # Handle axis formatting
        x_fmt = viz_plan.get("x_axis_format", {})
        y_fmt = viz_plan.get("y_axis_format", {})

        fig.update_layout(
            xaxis=dict(
                title=x_fmt.get("label", x_col),
                tickformat=x_fmt.get("tick_format")
            ),
            yaxis=dict(
                title=y_fmt.get("label", "Count"),
                tickformat=y_fmt.get("tick_format", "~s")
            )
        )

        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, viz_plan: Dict[str, Any]) -> Optional[go.Figure]:
        """Create box plot"""
        y_col = viz_plan.get('y_column')
        x_col = viz_plan.get('x_column')
        title = viz_plan.get('title', 'Box Plot')

        # Clean and validate data
        df = self._clean_and_validate_data(df, x_col, y_col)
        
        if df.empty:
            print("[WARN] No valid data for box plot after cleaning")
            return None

        print("[DEBUG] Data used for box plot:")
        print(df[[x_col, y_col]] if x_col else df[[y_col]])

        fig = px.box(
            df,
            x=x_col if x_col and x_col in df.columns else None,
            y=y_col,
            title=title,
            template="plotly_white"
        )

        x_fmt = viz_plan.get("x_axis_format", {})
        y_fmt = viz_plan.get("y_axis_format", {})

        fig.update_layout(
            xaxis=dict(title=x_fmt.get("label", x_col), tickformat=x_fmt.get("tick_format")),
            yaxis=dict(title=y_fmt.get("label", y_col), tickformat=y_fmt.get("tick_format"))
        )

        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, viz_plan: Dict[str, Any]) -> Optional[go.Figure]:
        """Create pie chart"""
        x_col = viz_plan.get('x_column')
        title = viz_plan.get('title', 'Pie Chart')

        if not x_col or x_col not in df.columns:
            return None

        # Clean data
        df = df.dropna(subset=[x_col])
        
        if df.empty:
            print("[WARN] No valid data for pie chart after cleaning")
            return None

        counts = df[x_col].value_counts()

        print("[DEBUG] Data used for pie chart:")
        print(counts.head())

        fig = px.pie(
            names=counts.index,
            values=counts.values,
            title=title,
            template="plotly_white"
        )

        y_fmt = viz_plan.get("y_axis_format", {})
        hover_unit = y_fmt.get("unit", "")
        fig.update_traces(
            hovertemplate=f"%{{label}}: %{{value}} {hover_unit} (%{{percent}})<extra></extra>"
        )

        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, viz_plan: Dict[str, Any]) -> Optional[go.Figure]:
        """Create heatmap/correlation matrix"""
        title = viz_plan.get('title', 'Heatmap')

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("[WARN] No numeric columns found for heatmap")
            return None

        # Remove columns with all NaN values
        numeric_df = numeric_df.dropna(axis=1, how='all')
        
        if numeric_df.empty:
            print("[WARN] No valid numeric data for heatmap after cleaning")
            return None

        print("[DEBUG] Correlation matrix input:")
        print(numeric_df.head())

        corr_matrix = numeric_df.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title=title,
            template="plotly_white",
            color_continuous_scale='RdBu_r'
        )

        fig.update_layout(
            xaxis=dict(title="Variables"),
            yaxis=dict(title="Variables")
        )

        return fig

    def _create_correlation_matrix(self, df, viz_plan):
        """Create correlation matrix (alias for heatmap)"""
        return self._create_heatmap(df, viz_plan)
    
    def create_summary_dashboard(self, structured_data: Dict[str, pd.DataFrame]) -> Optional[go.Figure]:
        """Create a summary dashboard with multiple charts"""
        try:
            if not structured_data:
                return None
            
            # Get the first dataset
            first_dataset = list(structured_data.values())[0]
            numeric_cols = first_dataset.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = first_dataset.select_dtypes(include=['object']).columns.tolist()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Distribution', 'Correlation Matrix', 'Top Categories', 'Summary Stats'),
                specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                       [{"type": "bar"}, {"type": "table"}]]
            )
            
            # Add histogram for first numeric column
            if numeric_cols:
                fig.add_trace(
                    go.Histogram(x=first_dataset[numeric_cols[0]], name=numeric_cols[0]),
                    row=1, col=1
                )
            
            # Add correlation heatmap
            if len(numeric_cols) > 1:
                corr_matrix = first_dataset[numeric_cols].corr()
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu'
                    ),
                    row=1, col=2
                )
            
            # Add bar chart for categorical data
            if categorical_cols:
                value_counts = first_dataset[categorical_cols[0]].value_counts().head(10)
                fig.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values),
                    row=2, col=1
                )
            
            # Add summary table
            if numeric_cols:
                summary_stats = first_dataset[numeric_cols].describe()
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Statistic'] + list(summary_stats.columns)),
                        cells=dict(values=[summary_stats.index] + [summary_stats[col] for col in summary_stats.columns])
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Data Summary Dashboard",
                showlegend=False,
                height=800,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating summary dashboard: {str(e)}")
            return None