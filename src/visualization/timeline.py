"""
src.visualization.timeline — Patient Event Timeline Visualization.

Plots a single patient's clinical events on a timeline, color-coded
by event type (LAB, VITAL, MEDICATION, PROCEDURE, DIAGNOSIS, CLINICAL_CONCEPT).
"""

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.data.canonical import PatientRecord


# Color palette for event types
EVENT_COLORS = {
    "LAB":              "#2196F3",  # Blue
    "VITAL":            "#4CAF50",  # Green
    "MEDICATION":       "#FF9800",  # Orange
    "PROCEDURE":        "#9C27B0",  # Purple
    "DIAGNOSIS":        "#F44336",  # Red
    "CLINICAL_CONCEPT": "#00BCD4",  # Teal
}

# Y-axis positions for event types
EVENT_Y_POSITIONS = {
    "VITAL":            5,
    "LAB":              4,
    "MEDICATION":       3,
    "PROCEDURE":        2,
    "DIAGNOSIS":        1,
    "CLINICAL_CONCEPT": 0,
}


def plot_patient_timeline(
    patient: PatientRecord,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 6),
) -> plt.Figure:
    """Plot a single patient's event timeline.

    Events are positioned by timestamp (x-axis) and event type (y-axis),
    with color coding by category. Numerical values are shown as marker
    sizes for labs and vitals.

    Parameters
    ----------
    patient : PatientRecord
        Patient record with clinical events.
    save_path : str, optional
        If provided, save the figure to this path.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Group events by type
    for event in patient.events:
        x = event.timestamp
        y = EVENT_Y_POSITIONS.get(event.event_type, 0)
        color = EVENT_COLORS.get(event.event_type, "#999999")

        # Marker size: larger for events with values
        if event.value is not None:
            size = 40
        else:
            size = 25

        # Add slight y-jitter to prevent overlap within same type
        y_jitter = np.random.normal(0, 0.08)

        ax.scatter(
            x, y + y_jitter,
            c=color, s=size, alpha=0.7,
            edgecolors="white", linewidth=0.5,
            zorder=3,
        )

        # Annotate selected events with concept name
        if event.event_type in ("MEDICATION", "PROCEDURE", "DIAGNOSIS"):
            ax.annotate(
                event.concept_name,
                (x, y + y_jitter),
                fontsize=6, alpha=0.6,
                textcoords="offset points",
                xytext=(3, 5),
                rotation=30,
            )

    # Formatting
    ax.set_yticks(list(EVENT_Y_POSITIONS.values()))
    ax.set_yticklabels(list(EVENT_Y_POSITIONS.keys()), fontsize=10)
    ax.set_xlabel("Hours relative to t0 (Sepsis-3 recognition)", fontsize=11)
    ax.set_title(
        f"Patient {patient.patient_id} — Event Timeline  "
        f"(Mortality: {'Yes' if patient.mortality else 'No'}, "
        f"Events: {len(patient.events)})",
        fontsize=12,
        fontweight="bold",
    )

    # t0 line
    ax.axvline(x=0, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.7, label="t0")

    # Legend
    legend_patches = [
        mpatches.Patch(color=color, label=etype)
        for etype, color in EVENT_COLORS.items()
    ]
    legend_patches.append(
        plt.Line2D([0], [0], color="#E53935", linestyle="--", linewidth=1.5, label="t0")
    )
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        ncol=2,
    )

    # Grid
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax.set_ylim(-0.8, 5.8)

    # Disclaimer
    fig.text(
        0.5, 0.01,
        "Research demonstration using patient data. ",
        
        ha="center", fontsize=7, fontstyle="italic", color="#888888",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
