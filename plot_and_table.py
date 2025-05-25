import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from main import evaluate_function, preprocess_equation


class PlotAndTable:
    def __init__(self, parent, show_root_labels=True):
        self.parent = parent
        self.show_root_labels = show_root_labels  # Flag to control root value labels
        self.setup_ui()
        self.current_method = "incremental"  # Default method

    def setup_ui(self):
        # Create main container
        self.main_container = tk.Frame(self.parent)
        self.main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Result label at top
        self.result_label = tk.Label(
            self.main_container,
            text="",
            font=("Arial", 12, "bold"),
            fg="#2980b9",  # Blue
            anchor="w",
            justify="left",
            wraplength=1000
        )

        self.result_label.pack(fill="x", pady=(0, 10))

        # Equation interpretation label
        self.equation_label = tk.Label(
            self.main_container,
            text="",
            font=("Arial", 10),
            fg="#1e7e34",  # Green
            anchor="w",
            justify="left",
            wraplength=1000
        )
        self.equation_label.pack(fill="x", pady=(0, 5))

        # Create plot and table container
        content_frame = tk.Frame(self.main_container)
        content_frame.pack(fill="both", expand=True)

        # Plot frame
        plot_frame = tk.Frame(content_frame)
        plot_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.set_facecolor('#f0f0f0')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_xlabel("x", fontsize=10)
        self.ax.set_ylabel("f(x)", fontsize=10)
        self.ax.set_title("Function Plot", fontsize=12)

        # Create canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        # Table frame
        table_frame = tk.Frame(content_frame)
        table_frame.pack(fill="both", expand=True)

        # Create scrollable table
        self.tree_scroll_y = ttk.Scrollbar(table_frame, orient="vertical")
        self.tree_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")

        # Create treeview for each method
        self.create_incremental_table(table_frame)
        self.create_bisection_table(table_frame)
        self.create_newton_raphson_table(table_frame)
        self.create_regula_falsi_table(table_frame)
        self.create_secant_table(table_frame)

        # Show default table
        self.show_table("incremental")

    def create_incremental_table(self, parent):
        self.incremental_tree = ttk.Treeview(
            parent,
            columns=("Iteration", "Xl", "Δx", "Xu", "f(Xl)", "f(Xu)", "Product", "Remark"),
            show="headings",
            height=8,
            yscrollcommand=self.tree_scroll_y.set,
            xscrollcommand=self.tree_scroll_x.set
        )

        # Configure columns
        columns = [
            ("Iteration", "Iteration", 80, "center"),
            ("Xl", "Xl", 100, "center"),
            ("Δx", "Δx", 100, "center"),
            ("Xu", "Xu", 100, "center"),
            ("f(Xl)", "f(Xl)", 120, "center"),
            ("f(Xu)", "f(Xu)", 120, "center"),
            ("Product", "Product", 80, "center"),
            ("Remark", "Remark", 150, "center")
        ]

        for col_id, heading, width, anchor in columns:
            self.incremental_tree.heading(col_id, text=heading)
            self.incremental_tree.column(col_id, width=width, anchor=anchor)

    def create_bisection_table(self, parent):
        self.bisection_tree = ttk.Treeview(
            parent,
            columns=("Iteration", "Xl", "Xr", "Xu", "f(Xl)", "f(Xr)", "|Ea|,%", "f(Xl).f(Xr)", "Remark"),
            show="headings",
            height=8,
            yscrollcommand=self.tree_scroll_y.set,
            xscrollcommand=self.tree_scroll_x.set
        )

        # Configure columns
        columns = [
            ("Iteration", "Iteration", 80, "center"),
            ("Xl", "Xl", 100, "center"),
            ("Xr", "Xr", 100, "center"),
            ("Xu", "Xu", 100, "center"),
            ("f(Xl)", "f(Xl)", 120, "center"),
            ("f(Xr)", "f(Xr)", 120, "center"),
            ("|Ea|,%", "|Ea|,%", 100, "center"),
            ("f(Xl).f(Xr)", "f(Xl).f(Xr)", 100, "center"),
            ("Remark", "Remark", 150, "center")
        ]

        for col_id, heading, width, anchor in columns:
            self.bisection_tree.heading(col_id, text=heading)
            self.bisection_tree.column(col_id, width=width, anchor=anchor)

    def create_newton_raphson_table(self, parent):
        self.newton_raphson_tree = ttk.Treeview(
            parent,
            columns=("Iteration", "Xi", "|Ea|,%", "f(Xi)", "f'(Xi)", "Remark"),
            show="headings",
            height=8,
            yscrollcommand=self.tree_scroll_y.set,
            xscrollcommand=self.tree_scroll_x.set
        )

        # Configure columns
        columns = [
            ("Iteration", "Iteration", 80, "center"),
            ("Xi", "Xi", 100, "center"),
            ("|Ea|,%", "|Ea|,%", 100, "center"),
            ("f(Xi)", "f(Xi)", 120, "center"),
            ("f'(Xi)", "f'(Xi)", 120, "center"),
            ("Remark", "Remark", 150, "center")
        ]

        for col_id, heading, width, anchor in columns:
            self.newton_raphson_tree.heading(col_id, text=heading)
            self.newton_raphson_tree.column(col_id, width=width, anchor=anchor)

    def create_regula_falsi_table(self, parent):
        self.regula_falsi_tree = ttk.Treeview(
            parent,
            columns=("Iteration", "XL", "XU", "XR", "|Ea|,%", "f(XL)", "f(XU)", "f(XR)", "f(XL)*f(XR)", "Remark"),
            show="headings",
            height=8,
            yscrollcommand=self.tree_scroll_y.set,
            xscrollcommand=self.tree_scroll_x.set
        )

        # Configure columns
        columns = [
            ("Iteration", "Iteration", 80, "center"),
            ("XL", "XL", 100, "center"),
            ("XU", "XU", 100, "center"),
            ("XR", "XR", 100, "center"),
            ("|Ea|,%", "|Ea|,%", 100, "center"),
            ("f(XL)", "f(XL)", 120, "center"),
            ("f(XU)", "f(XU)", 120, "center"),
            ("f(XR)", "f(XR)", 120, "center"),
            ("f(XL)*f(XR)", "f(XL)*f(XR)", 100, "center"),
            ("Remark", "Remark", 150, "center")
        ]

        for col_id, heading, width, anchor in columns:
            self.regula_falsi_tree.heading(col_id, text=heading)
            self.regula_falsi_tree.column(col_id, width=width, anchor=anchor)

    def create_secant_table(self, parent):
        self.secant_tree = ttk.Treeview(
            parent,
            columns=("Iteration", "Xi-1", "Xi", "Xi+1", "|Ea|,%", "f(Xi-1)", "f(Xi)", "f(Xi+1)", "Remark"),
            show="headings",
            height=8,
            yscrollcommand=self.tree_scroll_y.set,
            xscrollcommand=self.tree_scroll_x.set
        )

        # Configure columns
        columns = [
            ("Iteration", "Iteration", 80, "center"),
            ("Xi-1", "Xi-1", 100, "center"),
            ("Xi", "Xi", 100, "center"),
            ("Xi+1", "Xi+1", 100, "center"),
            ("|Ea|,%", "|Ea|,%", 100, "center"),
            ("f(Xi-1)", "f(Xi-1)", 120, "center"),
            ("f(Xi)", "f(Xi)", 120, "center"),
            ("f(Xi+1)", "f(Xi+1)", 120, "center"),
            ("Remark", "Remark", 150, "center")
        ]

        for col_id, heading, width, anchor in columns:
            self.secant_tree.heading(col_id, text=heading)
            self.secant_tree.column(col_id, width=width, anchor=anchor)

    def show_table(self, method):
        self.current_method = method

        # Hide all tables first
        if hasattr(self, 'incremental_tree'):
            self.incremental_tree.pack_forget()
        if hasattr(self, 'bisection_tree'):
            self.bisection_tree.pack_forget()
        if hasattr(self, 'newton_raphson_tree'):
            self.newton_raphson_tree.pack_forget()
        if hasattr(self, 'regula_falsi_tree'):
            self.regula_falsi_tree.pack_forget()
        if hasattr(self, 'secant_tree'):
            self.secant_tree.pack_forget()

        # Configure scrollbars
        self.tree_scroll_y.pack(side="right", fill="y")
        self.tree_scroll_x.pack(side="bottom", fill="x")

        # Show the appropriate table
        if method == "incremental":
            self.tree = self.incremental_tree
            self.tree.pack(side="left", fill="both", expand=True)
            self.tree_scroll_y.config(command=self.incremental_tree.yview)
            self.tree_scroll_x.config(command=self.incremental_tree.xview)
        elif method == "bisection":
            self.tree = self.bisection_tree
            self.tree.pack(side="left", fill="both", expand=True)
            self.tree_scroll_y.config(command=self.bisection_tree.yview)
            self.tree_scroll_x.config(command=self.bisection_tree.xview)
        elif method == "newton_raphson":
            self.tree = self.newton_raphson_tree
            self.tree.pack(side="left", fill="both", expand=True)
            self.tree_scroll_y.config(command=self.newton_raphson_tree.yview)
            self.tree_scroll_x.config(command=self.newton_raphson_tree.xview)
        elif method == "regula_falsi":
            self.tree = self.regula_falsi_tree
            self.tree.pack(side="left", fill="both", expand=True)
            self.tree_scroll_y.config(command=self.regula_falsi_tree.yview)
            self.tree_scroll_x.config(command=self.regula_falsi_tree.xview)
        elif method == "secant":
            self.tree = self.secant_tree
            self.tree.pack(side="left", fill="both", expand=True)
            self.tree_scroll_y.config(command=self.secant_tree.yview)
            self.tree_scroll_x.config(command=self.secant_tree.xview)

    def update_result_label(self, text):
        self.result_label.config(text=text)

    def update_equation_label(self, original_eqn):
        """Show how the equation is interpreted"""
        if original_eqn:
            processed_eqn = preprocess_equation(original_eqn)
            self.equation_label.config(text=f"Interpreted as: {processed_eqn}")
        else:
            self.equation_label.config(text="")

    def update_plot(self, points, eqn, roots=None):
        self.ax.clear()

        if points and len(points) > 0:
            x_vals, y_vals = zip(*points)

            # Plot evaluation points
            self.ax.scatter(x_vals, y_vals, color='#ff9800', s=30, label='Evaluation points')  # Orange

            # Focus on the area around the roots if available
            if roots and len(roots) > 0:
                # Get the range containing all roots
                root_min = min(roots) if isinstance(roots, list) else roots
                root_max = max(roots) if isinstance(roots, list) else roots

                # Add padding around the roots
                padding = max((root_max - root_min) * 0.5, 2.0)
                x_focus_min = root_min - padding
                x_focus_max = root_max + padding
            else:
                # If no roots, use a reasonable range around the evaluation points
                x_focus_min = min(x_vals)
                x_focus_max = max(x_vals)

            # Create smooth curve for the function with focus on the interesting area
            x_cont = np.linspace(x_focus_min - 1, x_focus_max + 1, 400)
            y_cont = []
            x_filtered = []

            # Filter out extreme values that would distort the plot
            for x in x_cont:
                try:
                    y = evaluate_function(x, eqn)
                    # Only include points with reasonable y values
                    if abs(y) < 100:  # Adjust this threshold as needed
                        y_cont.append(y)
                        x_filtered.append(x)
                except Exception:
                    continue

            if x_filtered and y_cont:
                # Plot function curve with filtered points
                self.ax.plot(x_filtered, y_cont, '#2980b9', linewidth=2, label=f'f(x) = {eqn}')  # Blue
                self.ax.axhline(0, color='black', linewidth=1)

                # For secant method, plot the secant lines between iterations
                if self.current_method == "secant" and len(points) >= 2:
                    # Only plot secant lines for points that are within our view
                    visible_points = []
                    for point in points:
                        x, y = point
                        if x_focus_min - 5 <= x <= x_focus_max + 5 and abs(y) < 100:
                            visible_points.append(point)

                    # Plot secant lines between consecutive visible points
                    for i in range(len(visible_points) - 1):
                        x_secant = [visible_points[i][0], visible_points[i + 1][0]]
                        y_secant = [visible_points[i][1], visible_points[i + 1][1]]
                        self.ax.plot(x_secant, y_secant, '#1e7e34', linestyle='--', alpha=0.5, linewidth=1)  # Green

                # Plot roots if found
                if roots:
                    if isinstance(roots, list):
                        for root in roots:
                            # Add vertical line at root
                            self.ax.axvline(x=root, color='#1e7e34', linestyle='--', alpha=0.7)  # Green
                            # Plot the root point
                            self.ax.plot(root, 0, 'go', markersize=8)  # Green dot for root

                            # Add label with root value if enabled
                            if self.show_root_labels:
                                # Position the label slightly above the x-axis
                                self.ax.annotate(f'x = {root:.6f}',
                                                 xy=(root, 0),
                                                 xytext=(root, 0.5),  # Adjust the y-offset as needed
                                                 ha='center',
                                                 va='bottom',
                                                 bbox=dict(boxstyle='round,pad=0.5', fc='#ff9800', alpha=0.7),
                                                 # Orange box
                                                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                                                 color='#ff9800'))  # Orange arrow
                    else:
                        # Add vertical line at root
                        self.ax.axvline(x=roots, color='#1e7e34', linestyle='--', alpha=0.7)  # Green
                        # Plot the root point
                        self.ax.plot(roots, 0, 'go', markersize=8)  # Green dot for root

                        # Add label with root value if enabled
                        if self.show_root_labels:
                            # Position the label slightly above the x-axis
                            self.ax.annotate(f'x = {roots:.6f}',
                                             xy=(roots, 0),
                                             xytext=(roots, 0.5),  # Adjust the y-offset as needed
                                             ha='center',
                                             va='bottom',
                                             bbox=dict(boxstyle='round,pad=0.5', fc='#ff9800', alpha=0.7),  # Orange box
                                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                                             color='#ff9800'))  # Orange arrow

                # Set limits with reasonable y-range
                y_min = min(y_cont)
                y_max = max(y_cont)
                y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.5

                self.ax.set_xlim(x_focus_min - 1, x_focus_max + 1)
                self.ax.set_ylim(y_min - y_padding, y_max + y_padding)

                # Add grid for better readability
                self.ax.grid(True, linestyle='--', alpha=0.7)

                # Set labels
                self.ax.set_xlabel("x", fontsize=10)
                self.ax.set_ylabel("f(x)", fontsize=10)
                self.ax.set_title("Function Plot", fontsize=12)

                # Remove duplicate legend entries
                handles, labels = self.ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')

            self.canvas.draw()

    def update_table(self, table_data):
        # Clear current table
        for row in self.tree.get_children():
            self.tree.delete(row)

        if table_data:
            # Insert data based on current method
            if self.current_method == "incremental":
                for row in table_data:
                    self.tree.insert("", "end", values=(
                        row.get("Iteration", ""),
                        row.get("Xl", ""),
                        row.get("Δx", ""),
                        row.get("Xu", ""),
                        row.get("f(Xl)", ""),
                        row.get("f(Xu)", ""),
                        row.get("Product", ""),
                        row.get("Remark", "")
                    ))
            elif self.current_method == "bisection":
                for i, row in enumerate(table_data):
                    item_id = self.tree.insert("", "end", values=(
                        row.get("Iteration", ""),
                        row.get("Xl", ""),
                        row.get("Xr", ""),
                        row.get("Xu", ""),
                        row.get("f(Xl)", ""),
                        row.get("f(Xr)", ""),
                        row.get("|Ea|,%", ""),
                        row.get("f(Xl).f(Xr)", ""),
                        row.get("Remark", "")
                    ))

                    # Color-code rows based on subinterval
                    if "1st subinterval" in row.get("Remark", ""):
                        self.tree.item(item_id, tags=("first_subinterval",))
                    elif "2nd subinterval" in row.get("Remark", ""):
                        self.tree.item(item_id, tags=("second_subinterval",))

                # Configure tag colors - using green and blue
                self.tree.tag_configure("first_subinterval", background="#d1e7dd")  # Light green
                self.tree.tag_configure("second_subinterval", background="#e8f4ff")  # Light blue
            elif self.current_method == "newton_raphson":
                for row in table_data:
                    self.tree.insert("", "end", values=(
                        row.get("Iteration", ""),
                        row.get("Xi", ""),
                        row.get("|Ea|,%", ""),
                        row.get("f(Xi)", ""),
                        row.get("f'(Xi)", ""),
                        row.get("Remark", "")
                    ))
            elif self.current_method == "regula_falsi":
                for row in table_data:
                    self.tree.insert("", "end", values=(
                        row.get("Iteration", ""),
                        row.get("XL", ""),
                        row.get("XU", ""),
                        row.get("XR", ""),
                        row.get("|Ea|,%", ""),
                        row.get("f(XL)", ""),
                        row.get("f(XU)", ""),
                        row.get("f(XR)", ""),
                        row.get("f(XL)*f(XR)", ""),
                        row.get("Remark", "")
                    ))
            elif self.current_method == "secant":
                for row in table_data:
                    self.tree.insert("", "end", values=(
                        row.get("Iteration", ""),
                        row.get("Xi-1", ""),
                        row.get("Xi", ""),
                        row.get("Xi+1", ""),
                        row.get("|Ea|,%", ""),
                        row.get("f(Xi-1)", ""),
                        row.get("f(Xi)", ""),
                        row.get("f(Xi+1)", ""),
                        row.get("Remark", "")
                    ))

    def clear_all(self):
        self.ax.clear()
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title("Function Plot", fontsize=12)
        self.ax.set_xlabel("x", fontsize=10)
        self.ax.set_ylabel("f(x)", fontsize=10)
        self.canvas.draw()
        self.update_table([])
        self.update_result_label("")
        self.update_equation_label("")