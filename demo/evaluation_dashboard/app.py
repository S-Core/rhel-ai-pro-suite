import sys
import yaml

import altair as alt
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from elasticsearch import Elasticsearch, ConnectionError, NotFoundError
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional


class ElasticsearchConfig(BaseModel):
    hosts: List[str]
    user: Optional[str] = None
    password: Optional[str] = None
    ca_certs: Optional[str] = None


class VectorStoreConfig(BaseModel):
    elasticsearch: ElasticsearchConfig


class EvaluationMetricsConfig(BaseModel):
    testset_index_name: str = Field(..., alias="testset_index_name")
    evaluation_index_name: str = Field(..., alias="evaluation_index_name")
    metrics: List[str]


class PluginsConfig(BaseModel):
    vector_store: VectorStoreConfig
    evaluation: dict[str, EvaluationMetricsConfig]


class AppConfig(BaseModel):
    plugins: PluginsConfig


class Config:
    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path("./config/configuration.yml")

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        self.config = AppConfig(**config_dict)


class DashboardApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.setup_page_config()
        self.setup_styling()
        self.es_client = None
        self.df = None
        self.validate_elasticsearch_connection()
        if self.es_client:
            self.validate_indices()
            self.df = self.load_data_from_elasticsearch()

    def setup_page_config(self):
        st.set_page_config(
            page_title="RHEL AI Evaluation Dashboard", page_icon="📊", layout="wide"
        )
        alt.themes.enable("dark")

    def setup_styling(self):
        st.markdown(
            """
            <style>
            [data-testid="block-container"] {
                padding-left: 2rem;
                padding-right: 2rem;
                padding-top: 1rem;
                padding-bottom: 0rem;
                margin-bottom: -7rem;
            }

            [data-testid="stVerticalBlock"] {
                padding-left: 0rem;
                padding-right: 0rem;
            }

            [data-testid="stMetric"] {
                background-color: #393939;
                text-align: center;
                padding: 15px 0;
            }

            [data-testid="stMetricLabel"] {
                display: flex;
                justify-content: center;
                align-items: center;
            }

            div[data-testid="stHorizontalBlock"] > div {
                width: fit-content !important;
                flex-grow: 0 !important;
                padding-right: 2rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def validate_elasticsearch_connection(self):
        """Validates the Elasticsearch connection and displays a warning if connection fails."""
        es_config = self.config.plugins.vector_store.elasticsearch
        client_kwargs = {
            "hosts": es_config.hosts[0],
            "request_timeout": 60,  # Set timeout to 60 seconds
            "verify_certs": False,
        }

        if es_config.user is not None and es_config.password is not None:
            client_kwargs["basic_auth"] = (es_config.user, es_config.password)

        if es_config.ca_certs is not None:
            client_kwargs["ca_certs"] = str(Path(es_config.ca_certs))

        try:
            self.es_client = Elasticsearch(**client_kwargs)
            if not self.es_client.ping():
                st.error(
                    "⚠️ Cannot connect to Elasticsearch. Please check if the server is running."
                )
                return False
        except ConnectionError:
            st.error(
                "⚠️ Failed to connect to Elasticsearch. Please check your connection settings."
            )
            return False
        return True

    def validate_indices(self):
        """Checks if required indices exist and displays warnings for missing indices."""
        if not self.es_client:
            return False

        eval_config = next(iter(self.config.plugins.evaluation.values()))
        missing_indices = []

        try:
            if not self.es_client.indices.exists(index=eval_config.testset_index_name):
                missing_indices.append(eval_config.testset_index_name)
            if not self.es_client.indices.exists(
                index=eval_config.evaluation_index_name
            ):
                missing_indices.append(eval_config.evaluation_index_name)

            if missing_indices:
                print(
                    f"⚠️ The following indices were not found: {', '.join(missing_indices)}"
                )
                return False
        except Exception as e:
            st.error(f"⚠️ Error occurred while checking indices: {str(e)}")
            return False
        return True

    @st.cache_data
    def load_data_from_elasticsearch(_self):
        """Loads data from Elasticsearch and handles potential errors."""
        if not _self.es_client:
            return pd.DataFrame()  # Return empty DataFrame

        eval_config = next(iter(_self.config.plugins.evaluation.values()))
        query = {"query": {"match_all": {}}, "size": 10000}

        try:
            response = _self.es_client.search(
                index=eval_config.evaluation_index_name, body=query
            )

        except NotFoundError:
            print(f"⚠️ Index {eval_config.evaluation_index_name} not found.")
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ Error occurred while loading data: {str(e)}")
            return pd.DataFrame()

        records = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            record = {
                "domain": source["metadata"]["domain"],
                "faithfulness": source["scores"]["faithfulness"],
                "answer_relevancy": source["scores"]["answerRelevancy"],
                "context_precision": source["scores"]["contextPrecision"],
                "context_recall": source["scores"]["contextRecall"],
                "context_entities_recall": source["scores"]["contextEntityRecall"],
                "noise_sensitivity": source["scores"]["noiseSensitivity"],
                "answer_similarity": source["scores"]["answerSimilarity"],
            }
            records.append(record)

        return pd.DataFrame(records)

    def render_header(self):
        st.title("📊 RHEL AI Evaluation Dashboard")

    def render_domain_overview(self):
        st.markdown("### Domain Overview")
        main_cols = st.columns(2)
        with main_cols[0]:
            st.metric("Total Domains", len(self.df["domain"].unique()))
        with main_cols[1]:
            st.metric("Total Samples", len(self.df))

        metrics = [
            ("Faithfulness", "faithfulness"),
            ("Answer relevancy", "answer_relevancy"),
            ("Noise Sensitivity", "noise_sensitivity"),
            ("Context entities recall", "context_entities_recall"),
            ("Context Precision", "context_precision"),
            ("Context Recall", "context_recall"),
            ("Answer Similarity", "answer_similarity"),
        ]

        metric_cols = st.columns(len(metrics), gap="small")
        for col, (label, metric) in zip(metric_cols, metrics):
            with col:
                st.metric(f"Avg {label}", f"{self.df[metric].mean():.2f}")

    def render_detailed_analysis(self):
        st.markdown("### Detailed Analysis")
        col = st.columns(3, gap="small")

        metrics = [
            "faithfulness",
            "answer_relevancy",
            "answer_similarity",
            "context_entities_recall",
            "context_precision",
            "context_recall",
            "noise_sensitivity",
        ]

        with col[0]:
            self.render_box_plot(metrics)
        with col[1]:
            self.render_correlation_heatmap(metrics)
        with col[2]:
            self.render_radar_chart(metrics)

    def render_box_plot(self, metrics):
        st.markdown("#### Metric Distribution by Domain")
        selected_metric = st.selectbox(
            "Select Metric for Box Plot", metrics, key="box_plot"
        )

        fig = px.box(
            self.df,
            x="domain",
            y=selected_metric,
            title=f"Distribution of {selected_metric} by Domain",
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_correlation_heatmap(self, metrics):
        st.markdown("#### Metric Correlations")
        selected_metrics = st.multiselect(
            "Select Metrics for Correlation",
            metrics,
            default=metrics[:4],
            key="correlation",
        )

        if selected_metrics:
            corr = self.df[selected_metrics].corr()
            fig = px.imshow(
                corr,
                labels=dict(x="Metrics", y="Metrics", color="Correlation"),
                x=selected_metrics,
                y=selected_metrics,
                aspect="auto",
                color_continuous_scale="RdBu_r",
            )
            fig.update_traces(text=corr.values.round(2), texttemplate="%{text}")
            fig.update_layout(
                title="Correlation between Selected RAGAS Metrics",
                template="plotly_dark",
                plot_bgcolor="rgba(0, 0, 0, 0)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_radar_chart(self, metrics):
        st.markdown("#### Domain Comparison")
        selected_domains = st.multiselect(
            "Select Domains to Compare",
            list(self.df.domain.unique()),
            default=list(self.df.domain.unique())[:3],
            key="radar",
        )

        radar_data = self.df.groupby("domain")[metrics].mean()
        fig = go.Figure()

        for domain in selected_domains:
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_data.loc[domain], theta=metrics, name=domain, fill="toself"
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="RAGAS Metrics by Domain",
            template="plotly_dark",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_domain_performance(self):
        st.markdown("### Domain Performance Analysis")
        selected_domain = st.selectbox(
            "Select Domain for Statistics", list(self.df.domain.unique()), key="stats"
        )

        df_selected = self.df[self.df.domain == selected_domain]
        metrics = [col for col in self.df.columns if col != "domain"]

        metrics_summary = df_selected[metrics].describe()
        st.dataframe(metrics_summary, height=300, use_container_width=True)

        if st.checkbox("Show Raw Data"):
            st.write(df_selected)

    def render_footer(self):
        st.markdown(
            """
            ---
            ### About
            This dashboard visualizes [RAGAS](https://github.com/explodinggradients/ragas) evaluation metrics:
            - **Faithfulness**: Measures how well the answer aligns with the given context
            - **Answer Relevancy**: Evaluates if the answer is relevant to the question
            - **Context Entities Recall**: Measures how well the important entities from context are captured
            - **Context Precision/Recall**: Evaluates the precision and recall of context usage
            - **Context Utilization**: Measures how effectively the context is used
            - **Noise Sensitivity**: Evaluates robustness to noise in the context
            - **Answer Similarity**: Evaluates how closely a generated answer aligns with the ground truth
            """
        )

    def run(self):
        """Runs the dashboard with data validation."""
        self.render_header()

        # Only proceed with visualization if data is available
        if self.df is not None and not self.df.empty:
            self.render_domain_overview()
            self.render_detailed_analysis()
            self.render_domain_performance()
            self.render_footer()
        else:
            st.warning(
                "⚠️ Unable to load data. Please check Elasticsearch connection and indices."
            )


def main():
    """Main entry point"""
    try:
        if "--app-config" in sys.argv:
            config_path = Path(sys.argv[sys.argv.index("--app-config") + 1])
        else:
            config_path = None
        config = Config(config_path).config
        app = DashboardApp(config)
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
