import tkinter as tk
from tkinter import ttk, messagebox
from plot_and_table import PlotAndTable



from main import (
    incremental_search, graphical_method, bisection_method,
    newton_raphson_method, regula_falsi_method, secant_method,
    preprocess_equation, evaluate_function
)


class RootFinderApp:


    #Designs
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Methods Calculator")

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate window size (80% of screen size)
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)

        # Calculate position (center of screen)
        position_x = int((screen_width - window_width) / 2)
        position_y = int((screen_height - window_height) / 2)

        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        self.root.minsize(800, 600)  # Set minimum size to ensure usability

        # Make the window responsive
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Configure style
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('Sidebar.TFrame', background='#e8f4ff')  # Light blue background

        # Enhanced button style
        self.style.configure('Method.TButton',
                             font=('Arial', 10, 'bold'),
                             padding=8,
                             background='#d1e7dd',  # Light green
                             relief='flat')
        self.style.map('Method.TButton',
                       background=[('active', '#a3cfbb'), ('pressed', '#75b798')],  # Green shades
                       relief=[('pressed', 'sunken')])

        # Active method button style
        self.style.configure('Active.TButton',
                             font=('Arial', 10, 'bold'),
                             padding=8,
                             background='#1e7e34',  # Darker green
                             relief='sunken')

        # Menu button style
        self.style.configure('Menu.TButton',
                             font=('Arial', 9, 'bold'),
                             padding=5,
                             background='#d1e7dd')  # Light green
        self.style.map('Menu.TButton',
                       background=[('active', '#a3cfbb'), ('pressed', '#75b798')])  # Green shades

        # Active menu button style
        self.style.configure('ActiveMenu.TButton',
                             font=('Arial', 9, 'bold'),
                             padding=5,
                             background='#1e7e34')  # Darker green

        # Enhanced sidebar styles
        self.style.configure('Sidebar.TLabelframe', background='#e8f4ff')  # Light blue
        self.style.configure('Sidebar.TLabelframe.Label', background='#e8f4ff', font=('Arial', 10, 'bold'))

        # Enhanced notebook styles
        self.style.configure('Sidebar.TNotebook', background='#e8f4ff')  # Light blue
        self.style.configure('Sidebar.TNotebook.Tab',
                             font=('Arial', 8),
                             padding=[6, 2],
                             background='#d1e7dd')  # Light green
        self.style.map('Sidebar.TNotebook.Tab',
                       background=[('selected', '#1e7e34'), ('active', '#a3cfbb')],  # Green shades
                       foreground=[('selected', 'white')])

        # Create a custom header style
        self.style.configure('Header.TFrame', background='#2980b9')  # Blue
        self.style.configure('Header.TLabel',
                             font=('Arial', 12, 'bold'),
                             foreground='white',
                             background='#2980b9')  # Blue

        # Input panel style
        self.style.configure('Input.TFrame', background='#f5f5f5')
        self.style.configure('Input.TLabelframe', background='#f5f5f5')
        self.style.configure('Input.TLabelframe.Label', background='#f5f5f5', font=('Arial', 10, 'bold'))

        # Initialize method-specific function entries
        self.graph_function_entry = None
        self.incr_function_entry = None
        self.bisection_function_entry = None
        self.newton_function_entry = None
        self.regula_falsi_function_entry = None
        self.secant_function_entry = None

        # Create widgets
        self.create_widgets()

        # Bind resize event
        self.root.bind("<Configure>", self.on_window_resize)












    def on_window_resize(self, event):
        """Handle window resize events"""
        # Only respond to the root window's resize events
        if event.widget == self.root:
            # Update the sidebar width based on window width
            window_width = event.width
            sidebar_width = min(300, int(window_width * 0.25))  # 25% of window width, max 300px

            # Update sidebar width
            if hasattr(self, 'sidebar'):
                self.sidebar.configure(width=sidebar_width)

                # Update canvas width inside sidebar
                for child in self.sidebar.winfo_children():
                    if isinstance(child, tk.Canvas):
                        child.configure(width=sidebar_width - 20)
                        # Update the window width in the canvas
                        for item in child.find_all():
                            if child.type(item) == "window":
                                child.itemconfigure(item, width=sidebar_width - 20)

    def create_widgets(self):
        # Main container with grid layout for better responsiveness
        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky='nsew')

        # Configure main container to expand with window
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(0, weight=0)  # Header
        self.main_container.rowconfigure(1, weight=0)  # Method menu
        self.main_container.rowconfigure(2, weight=0)  # Input panel
        self.main_container.rowconfigure(3, weight=1)  # Content area

        # Create header
        self.create_header()

        # Create method menu
        self.create_method_menu()

        # Create input panel
        self.create_input_panel()

        # Bottom container - using grid for better control
        self.bottom_container = ttk.Frame(self.main_container)
        self.bottom_container.grid(row=3, column=0, sticky='nsew')

        # Configure grid columns with weight
        self.bottom_container.columnconfigure(0, weight=0)  # Sidebar doesn't expand horizontally
        self.bottom_container.columnconfigure(1, weight=1)  # Content area expands
        self.bottom_container.rowconfigure(0, weight=1)  # Both expand vertically

        # Sidebar with fixed width
        self.create_sidebar()

        # Content area
        self.content_area = ttk.Frame(self.bottom_container)
        self.content_area.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        # Configure content area to be responsive
        self.content_area.columnconfigure(0, weight=1)
        self.content_area.rowconfigure(0, weight=1)

        # Visual components
        self.visuals = PlotAndTable(self.content_area, show_root_labels=True)  # Enable root labels

        # Method frames (now hidden in the content area, will be shown when method is selected)
        self.method_frames = {
            "graphical": self.create_graphical_frame(),
            "incremental": self.create_incremental_frame(),
            "bisection": self.create_bisection_frame(),
            "newton_raphson": self.create_newton_raphson_frame(),
            "regula_falsi": self.create_regula_falsi_frame(),
            "secant": self.create_secant_frame()
        }

        # Show default method - AFTER all UI elements are created
        self.show_method("incremental")






    def create_header(self):
        """Create a professional header for the application"""
        header_frame = ttk.Frame(self.main_container, style='Header.TFrame', height=40)
        header_frame.grid(row=0, column=0, sticky='ew')

        # Configure header to expand horizontally
        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, weight=1)

        # Make the header fixed height
        header_frame.grid_propagate(False)

        # App title
        title_label = ttk.Label(header_frame,
                                text="Numerical Methods Calculator",
                                style='Header.TLabel')
        title_label.grid(row=0, column=0, sticky='w', padx=15)

        # Current method indicator
        self.method_indicator = ttk.Label(header_frame,
                                          text="",
                                          style='Header.TLabel')
        self.method_indicator.grid(row=0, column=1, sticky='e', padx=15)






    def create_method_menu(self):
        """Create a menu-like bar for method selection"""
        menu_frame = ttk.Frame(self.main_container, style='TFrame', height=40)
        menu_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)

        # Methods list
        methods = [
            ("Graphical Method", "graphical"),
            ("Incremental Search", "incremental"),
            ("Bisection Method", "bisection"),
            ("Newton-Raphson Method", "newton_raphson"),
            ("Regula Falsi Method", "regula_falsi"),
            ("Secant Method", "secant")
        ]

        # Create a dictionary to store button references
        self.method_buttons = {}

        # Create method buttons in a horizontal layout
        for i, (method_name, method_key) in enumerate(methods):
            btn = ttk.Button(menu_frame, text=method_name,
                             command=lambda key=method_key: self.show_method(key),
                             style='Menu.TButton')
            btn.grid(row=0, column=i, padx=2)
            self.method_buttons[method_key] = btn



    def create_input_panel(self):
        """Create a panel for common input fields at the top"""
        input_frame = ttk.Frame(self.main_container, style='Input.TFrame')
        input_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)

        # Configure the input frame to expand horizontally
        input_frame.columnconfigure(1, weight=1)

        # Common equation input
        ttk.Label(input_frame, text="Function f(x):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.common_function_entry = ttk.Entry(input_frame, width=40)
        self.common_function_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.common_function_entry.insert(0, "x^2 - 4")  # Default example

        # Preview button
        preview_btn = ttk.Button(input_frame, text="Preview",
                                 command=lambda: self.preview_equation(self.common_function_entry.get()),
                                 style='Method.TButton')
        preview_btn.grid(row=0, column=2, padx=5, pady=5)

        # Store the input frame for later access
        self.input_frame = input_frame

        # Create parameter frames for each method (will be shown/hidden as needed)
        self.param_frames = {}



        # Graphical method parameters
        graph_params = ttk.Frame(input_frame)
        ttk.Label(graph_params, text="Lower bound (a):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.graph_lower_entry = ttk.Entry(graph_params, width=10)
        self.graph_lower_entry.insert(0, "-5")
        self.graph_lower_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(graph_params, text="Upper bound (b):").grid(row=0, column=2, sticky='e', padx=5, pady=5)
        self.graph_upper_entry = ttk.Entry(graph_params, width=10)
        self.graph_upper_entry.insert(0, "5")
        self.graph_upper_entry.grid(row=0, column=3, sticky='w', padx=5, pady=5)

        compute_graph_btn = ttk.Button(graph_params, text="Plot Function",
                                       command=self.compute_graphical_method,
                                       style='Method.TButton')
        compute_graph_btn.grid(row=0, column=4, padx=10, pady=5)

        self.param_frames["graphical"] = graph_params

        # Initialize method-specific function entries
        self.graph_function_entry = ttk.Entry(graph_params)  # Hidden entry to store the function
        self.graph_function_entry.insert(0, "")








        # Incremental search parameters
        incr_params = ttk.Frame(input_frame)
        ttk.Label(incr_params, text="Initial x:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.incr_start_entry = ttk.Entry(incr_params, width=8)
        self.incr_start_entry.insert(0, "0")
        self.incr_start_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(incr_params, text="Step size:").grid(row=0, column=2, sticky='e', padx=5, pady=5)
        self.incr_step_entry = ttk.Entry(incr_params, width=8)
        self.incr_step_entry.insert(0, "0.5")
        self.incr_step_entry.grid(row=0, column=3, sticky='w', padx=5, pady=5)

        ttk.Label(incr_params, text="Tolerance:").grid(row=0, column=4, sticky='e', padx=5, pady=5)
        self.incr_tolerance_entry = ttk.Entry(incr_params, width=8)
        self.incr_tolerance_entry.insert(0, "0.0001")
        self.incr_tolerance_entry.grid(row=0, column=5, sticky='w', padx=5, pady=5)

        ttk.Label(incr_params, text="Max Iter:").grid(row=0, column=6, sticky='e', padx=5, pady=5)
        self.incr_max_iter_entry = ttk.Entry(incr_params, width=8)
        self.incr_max_iter_entry.insert(0, "100")
        self.incr_max_iter_entry.grid(row=0, column=7, sticky='w', padx=5, pady=5)

        compute_incr_btn = ttk.Button(incr_params, text="Find Roots",
                                      command=self.compute_incremental_search,
                                      style='Method.TButton')
        compute_incr_btn.grid(row=0, column=8, padx=10, pady=5)

        self.param_frames["incremental"] = incr_params

        # Initialize method-specific function entry
        self.incr_function_entry = ttk.Entry(incr_params)  # Hidden entry to store the function
        self.incr_function_entry.insert(0, "")









        # Bisection method parameters
        bisection_params = ttk.Frame(input_frame)
        ttk.Label(bisection_params, text="Lower bound:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.bisection_lower_entry = ttk.Entry(bisection_params, width=8)
        self.bisection_lower_entry.insert(0, "0")
        self.bisection_lower_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(bisection_params, text="Upper bound:").grid(row=0, column=2, sticky='e', padx=5, pady=5)
        self.bisection_upper_entry = ttk.Entry(bisection_params, width=8)
        self.bisection_upper_entry.insert(0, "1")
        self.bisection_upper_entry.grid(row=0, column=3, sticky='w', padx=5, pady=5)

        ttk.Label(bisection_params, text="Tolerance:").grid(row=0, column=4, sticky='e', padx=5, pady=5)
        self.bisection_tolerance_entry = ttk.Entry(bisection_params, width=8)
        self.bisection_tolerance_entry.insert(0, "0.0001")
        self.bisection_tolerance_entry.grid(row=0, column=5, sticky='w', padx=5, pady=5)

        ttk.Label(bisection_params, text="Error %:").grid(row=0, column=6, sticky='e', padx=5, pady=5)
        self.bisection_error_entry = ttk.Entry(bisection_params, width=8)
        self.bisection_error_entry.insert(0, "0.5")
        self.bisection_error_entry.grid(row=0, column=7, sticky='w', padx=5, pady=5)

        ttk.Label(bisection_params, text="Max Iter:").grid(row=0, column=8, sticky='e', padx=5, pady=5)
        self.bisection_max_iter_entry = ttk.Entry(bisection_params, width=8)
        self.bisection_max_iter_entry.insert(0, "100")
        self.bisection_max_iter_entry.grid(row=0, column=9, sticky='w', padx=5, pady=5)

        compute_bisection_btn = ttk.Button(bisection_params, text="Find Roots",
                                           command=self.compute_bisection_method,
                                           style='Method.TButton')
        compute_bisection_btn.grid(row=0, column=10, padx=10, pady=5)

        self.param_frames["bisection"] = bisection_params

        # Initialize method-specific function entry
        self.bisection_function_entry = ttk.Entry(bisection_params)  # Hidden entry to store the function
        self.bisection_function_entry.insert(0, "")

















        # Newton-Raphson method parameters
        newton_params = ttk.Frame(input_frame)
        ttk.Label(newton_params, text="Initial guess:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.newton_initial_entry = ttk.Entry(newton_params, width=8)
        self.newton_initial_entry.insert(0, "-0.5")
        self.newton_initial_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(newton_params, text="Tolerance:").grid(row=0, column=2, sticky='e', padx=5, pady=5)
        self.newton_tolerance_entry = ttk.Entry(newton_params, width=8)
        self.newton_tolerance_entry.insert(0, "0.0001")
        self.newton_tolerance_entry.grid(row=0, column=3, sticky='w', padx=5, pady=5)

        ttk.Label(newton_params, text="Error %:").grid(row=0, column=4, sticky='e', padx=5, pady=5)
        self.newton_error_entry = ttk.Entry(newton_params, width=8)
        self.newton_error_entry.insert(0, "0.5")
        self.newton_error_entry.grid(row=0, column=5, sticky='w', padx=5, pady=5)

        ttk.Label(newton_params, text="Max Iter:").grid(row=0, column=6, sticky='e', padx=5, pady=5)
        self.newton_max_iter_entry = ttk.Entry(newton_params, width=8)
        self.newton_max_iter_entry.insert(0, "100")
        self.newton_max_iter_entry.grid(row=0, column=7, sticky='w', padx=5, pady=5)

        compute_newton_btn = ttk.Button(newton_params, text="Find Roots",
                                        command=self.compute_newton_raphson_method,
                                        style='Method.TButton')
        compute_newton_btn.grid(row=0, column=8, padx=10, pady=5)

        self.param_frames["newton_raphson"] = newton_params

        # Initialize method-specific function entry
        self.newton_function_entry = ttk.Entry(newton_params)  # Hidden entry to store the function
        self.newton_function_entry.insert(0, "")












        # Regula Falsi method parameters
        regula_falsi_params = ttk.Frame(input_frame)
        ttk.Label(regula_falsi_params, text="Lower bound:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.regula_falsi_lower_entry = ttk.Entry(regula_falsi_params, width=8)
        self.regula_falsi_lower_entry.insert(0, "-0.5")
        self.regula_falsi_lower_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(regula_falsi_params, text="Upper bound:").grid(row=0, column=2, sticky='e', padx=5, pady=5)
        self.regula_falsi_upper_entry = ttk.Entry(regula_falsi_params, width=8)
        self.regula_falsi_upper_entry.insert(0, "0.5")
        self.regula_falsi_upper_entry.grid(row=0, column=3, sticky='w', padx=5, pady=5)

        ttk.Label(regula_falsi_params, text="Tolerance:").grid(row=0, column=4, sticky='e', padx=5, pady=5)
        self.regula_falsi_tolerance_entry = ttk.Entry(regula_falsi_params, width=8)
        self.regula_falsi_tolerance_entry.insert(0, "0.0001")
        self.regula_falsi_tolerance_entry.grid(row=0, column=5, sticky='w', padx=5, pady=5)

        ttk.Label(regula_falsi_params, text="Error %:").grid(row=0, column=6, sticky='e', padx=5, pady=5)
        self.regula_falsi_error_entry = ttk.Entry(regula_falsi_params, width=8)
        self.regula_falsi_error_entry.insert(0, "0.5")
        self.regula_falsi_error_entry.grid(row=0, column=7, sticky='w', padx=5, pady=5)

        ttk.Label(regula_falsi_params, text="Max Iter:").grid(row=0, column=8, sticky='e', padx=5, pady=5)
        self.regula_falsi_max_iter_entry = ttk.Entry(regula_falsi_params, width=8)
        self.regula_falsi_max_iter_entry.insert(0, "100")
        self.regula_falsi_max_iter_entry.grid(row=0, column=9, sticky='w', padx=5, pady=5)

        compute_regula_falsi_btn = ttk.Button(regula_falsi_params, text="Find Roots",
                                              command=self.compute_regula_falsi_method,
                                              style='Method.TButton')
        compute_regula_falsi_btn.grid(row=0, column=10, padx=10, pady=5)

        self.param_frames["regula_falsi"] = regula_falsi_params

        # Initialize method-specific function entry
        self.regula_falsi_function_entry = ttk.Entry(regula_falsi_params)  # Hidden entry to store the function
        self.regula_falsi_function_entry.insert(0, "")











        # Secant method parameters
        secant_params = ttk.Frame(input_frame)
        ttk.Label(secant_params, text="First guess:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.secant_x0_entry = ttk.Entry(secant_params, width=8)
        self.secant_x0_entry.insert(0, "-1")
        self.secant_x0_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(secant_params, text="Second guess:").grid(row=0, column=2, sticky='e', padx=5, pady=5)
        self.secant_x1_entry = ttk.Entry(secant_params, width=8)
        self.secant_x1_entry.insert(0, "0")
        self.secant_x1_entry.grid(row=0, column=3, sticky='w', padx=5, pady=5)

        ttk.Label(secant_params, text="Tolerance:").grid(row=0, column=4, sticky='e', padx=5, pady=5)
        self.secant_tolerance_entry = ttk.Entry(secant_params, width=8)
        self.secant_tolerance_entry.insert(0, "0.0001")
        self.secant_tolerance_entry.grid(row=0, column=5, sticky='w', padx=5, pady=5)

        ttk.Label(secant_params, text="Error %:").grid(row=0, column=6, sticky='e', padx=5, pady=5)
        self.secant_error_entry = ttk.Entry(secant_params, width=8)
        self.secant_error_entry.insert(0, "0.5")
        self.secant_error_entry.grid(row=0, column=7, sticky='w', padx=5, pady=5)

        ttk.Label(secant_params, text="Max Iter:").grid(row=0, column=8, sticky='e', padx=5, pady=5)
        self.secant_max_iter_entry = ttk.Entry(secant_params, width=8)
        self.secant_max_iter_entry.insert(0, "100")
        self.secant_max_iter_entry.grid(row=0, column=9, sticky='w', padx=5, pady=5)

        compute_secant_btn = ttk.Button(secant_params, text="Find Roots",
                                        command=self.compute_secant_method,
                                        style='Method.TButton')
        compute_secant_btn.grid(row=0, column=10, padx=10, pady=5)

        self.param_frames["secant"] = secant_params

        # Initialize method-specific function entry
        self.secant_function_entry = ttk.Entry(secant_params)  # Hidden entry to store the function
        self.secant_function_entry.insert(0, "")










    def create_sidebar(self):
        """Create a responsive sidebar that adjusts with window size"""
        # Create a frame for the sidebar with initial width
        sidebar_width = int(self.root.winfo_width() * 0.25)  # 25% of window width
        sidebar_width = min(300, max(200, sidebar_width))  # Between 200 and 300 pixels

        sidebar = ttk.Frame(self.bottom_container, style='Sidebar.TFrame', width=sidebar_width)
        sidebar.grid(row=0, column=0, sticky='ns')

        # Prevent the sidebar from resizing based on content
        sidebar.grid_propagate(False)

        # Create a canvas with scrollbar for the sidebar content
        canvas = tk.Canvas(sidebar, bg='#e8f4ff', highlightthickness=0)  # Light blue
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)

        # Place canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create a frame inside the canvas for the scrollable content
        scrollable_frame = ttk.Frame(canvas, style='Sidebar.TFrame')

        # Configure canvas scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=sidebar_width - 20)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mousewheel scrolling with better event handling
        def _on_mousewheel(event):
            # Check if mouse is over the canvas
            x, y = sidebar.winfo_pointerxy()
            widget_under_mouse = sidebar.winfo_containing(x, y)

            # Only scroll if mouse is over the canvas or its children
            if widget_under_mouse and (widget_under_mouse == canvas or
                                       canvas.winfo_ismapped() and
                                       widget_under_mouse.winfo_toplevel() == canvas.winfo_toplevel()):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind mousewheel events only when mouse is over the sidebar
        def _bind_mousewheel(event):
            self.root.bind_all("<MouseWheel>", _on_mousewheel)
            self.root.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            self.root.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        def _unbind_mousewheel(event):
            self.root.unbind_all("<MouseWheel>")
            self.root.unbind_all("<Button-4>")
            self.root.unbind_all("<Button-5>")

        sidebar.bind("<Enter>", _bind_mousewheel)
        sidebar.bind("<Leave>", _unbind_mousewheel)

        # Title
        title = tk.Label(scrollable_frame, text="Reference & Help",
                         font=('Arial', 14, 'bold'),
                         bg='#e8f4ff', pady=10)  # Light blue
        title.pack(fill='x')

        # Quick reference section
        quick_ref_frame = ttk.LabelFrame(scrollable_frame, text="Quick Reference", style='Sidebar.TLabelframe')
        quick_ref_frame.pack(fill='x', padx=10, pady=5)

        quick_ref_text = tk.Text(quick_ref_frame, height=8, width=20, font=('Arial', 8), wrap='word', bg='#f5f5f5')
        quick_ref_text.pack(fill='x', padx=5, pady=5)
        quick_ref_text.insert('1.0', """Common Functions:

x^2 (x squared)
e^(-x) (exponential)
sin(x) (sine)
ln(x) (natural log)
sqrt(x) (square root)
x^3-2*x-5 (polynomial)
cos(x) (cosine)
tan(x) (tangent)
pi (π constant)
e (Euler's number)""")
        quick_ref_text.config(state='disabled')

        # Create instructions tab in the sidebar
        instructions_frame = ttk.LabelFrame(scrollable_frame, text="Instructions", style='Sidebar.TLabelframe')
        instructions_frame.pack(fill='x', padx=10, pady=5, expand=True)

        # Create a notebook (tabbed interface) in the sidebar with scrollable tabs
        self.instructions_notebook = ttk.Notebook(instructions_frame, style='Sidebar.TNotebook')

        # Enable tab scrolling
        self.enable_notebook_scrolling(self.instructions_notebook)

        self.instructions_notebook.pack(fill='both', expand=True, padx=2, pady=5)

        # Create tabs for each method
        self.create_general_instructions_tab()
        self.create_graphical_instructions_tab()
        self.create_incremental_instructions_tab()
        self.create_bisection_instructions_tab()
        self.create_newton_raphson_instructions_tab()
        self.create_regula_falsi_instructions_tab()
        self.create_secant_instructions_tab()
        self.create_legends_tab()

        # Store the sidebar reference to access later
        self.sidebar = sidebar
        self.scrollable_frame = scrollable_frame
        self.sidebar_canvas = canvas

    def create_general_instructions_tab(self):
        tab = ttk.Frame(self.instructions_notebook)
        self.instructions_notebook.add(tab, text="General")

        # Create a text widget with scrollbar
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill='both', expand=True, padx=2, pady=2)

        # Add both vertical and horizontal scrollbars
        v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal")
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')

        text = tk.Text(text_frame, width=30, height=15, wrap='none',
                       yscrollcommand=v_scrollbar.set,
                       xscrollcommand=h_scrollbar.set,
                       font=('Arial', 8), bg='#f5f5f5')
        text.pack(side='left', fill='both', expand=True)
        v_scrollbar.config(command=text.yview)
        h_scrollbar.config(command=text.xview)

        # Add instructions
        text.insert('1.0', """
NUMERICAL METHODS CALCULATOR

This application helps you find roots of equations using different numerical methods.

EQUATION INPUT FORMATS:
• Basic operations: +, -, *, /, ^ (or **)
• Variables: Use 'x' as the variable
• Functions: sin(x), cos(x), tan(x), exp(x), ln(x), log(x), sqrt(x)
• Constants: e (Euler's number), pi

EXAMPLES:
• Polynomial: x^3 - 2*x - 5
• Exponential: e^(-x) - x
• Trigonometric: sin(x) - cos(x)
• Logarithmic: ln(x) - 1

TIPS:
• Use parentheses for clarity: e^(-x) is better than e^-x
• Preview your equation before calculation
• Different methods work better for different equations
• If one method fails, try another method
• Check that your function changes sign in the interval
""")
        text.config(state='disabled')  # Make read-only

    def create_graphical_instructions_tab(self):
        tab = ttk.Frame(self.instructions_notebook)
        self.instructions_notebook.add(tab, text="Graphical")

        # Create a text widget with scrollbar
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill='both', expand=True, padx=2, pady=2)

        # Add both vertical and horizontal scrollbars
        v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal")
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')

        text = tk.Text(text_frame, width=30, height=15, wrap='none',
                       yscrollcommand=v_scrollbar.set,
                       xscrollcommand=h_scrollbar.set,
                       font=('Arial', 8), bg='#f5f5f5')
        text.pack(side='left', fill='both', expand=True)
        v_scrollbar.config(command=text.yview)
        h_scrollbar.config(command=text.xview)

        # Add instructions
        text.insert('1.0', """
GRAPHICAL METHOD

The graphical method visually identifies where a function crosses the x-axis.

HOW IT WORKS:
1. The function is evaluated at many points in the specified range
2. The application looks for sign changes (where the function crosses zero)
3. Each sign change indicates a root
4. The method refines each root using bisection

PARAMETERS:
• Function f(x): The equation to find roots for
• Lower bound (a): The left endpoint of the search range
• Upper bound (b): The right endpoint of the search range

WHEN TO USE:
• When you want to visualize the function
• When you need to find multiple roots
• When you're not sure where the roots are located
• As a first step before using more precise methods

EXAMPLES:
• x^2 - 4 (roots at x = -2 and x = 2)
• sin(x) (roots at x = 0, π, 2π, etc.)
• e^x - 5 (one root where e^x equals 5)

TIPS:
• Use a wide range to find all roots
• The method may miss roots if they're very close together
• Zoom in on the plot to see roots more clearly
""")
        text.config(state='disabled')  # Make read-only

    def create_incremental_instructions_tab(self):
        tab = ttk.Frame(self.instructions_notebook)
        self.instructions_notebook.add(tab, text="Incremental")

        # Create a text widget with scrollbar
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill='both', expand=True, padx=2, pady=2)

        # Add both vertical and horizontal scrollbars
        v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal")
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')

        text = tk.Text(text_frame, width=30, height=15, wrap='none',
                       yscrollcommand=v_scrollbar.set,
                       xscrollcommand=h_scrollbar.set,
                       font=('Arial', 8), bg='#f5f5f5')
        text.pack(side='left', fill='both', expand=True)
        v_scrollbar.config(command=text.yview)
        h_scrollbar.config(command=text.xview)

        # Add instructions
        text.insert('1.0', """
INCREMENTAL SEARCH METHOD

The incremental search method systematically searches for sign changes in a function.

HOW IT WORKS:
1. Start at the initial x value
2. Take steps of size Δx in the x-direction
3. Check if the function changes sign between steps
4. When a sign change is detected, refine the search in that interval
5. Continue until the desired number of roots is found or max iterations reached

PARAMETERS:
• Function f(x): The equation to find roots for
• Initial x (Xl): Starting point for the search
• Step size (Δx): Distance between evaluation points
• Tolerance: How close to zero f(x) must be to be considered a root
• Max Iterations: Maximum number of steps to take

WHEN TO USE:
• When you need to find multiple roots
• When you don't know approximately where the roots are
• For functions that change sign at their roots

EXAMPLES:
• x^3 - x - 1 (one real root)
• x^4 - 5*x^2 + 4 (roots at x = -2, -1, 1, 2)
• e^(-x) - x (one root near x = 0.567)

TIPS:
• Use a smaller step size for more accuracy (but slower search)
• Choose a starting point that gives you the best chance to find all roots
• The method can find up to 5 roots by default
""")
        text.config(state='disabled')  # Make read-only

    def create_bisection_instructions_tab(self):
        tab = ttk.Frame(self.instructions_notebook)
        self.instructions_notebook.add(tab, text="Bisection")

        # Create a text widget with scrollbar
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill='both', expand=True, padx=2, pady=2)

        # Add both vertical and horizontal scrollbars
        v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal")
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')

        text = tk.Text(text_frame, width=30, height=15, wrap='none',
                       yscrollcommand=v_scrollbar.set,
                       xscrollcommand=h_scrollbar.set,
                       font=('Arial', 8), bg='#f5f5f5')
        text.pack(side='left', fill='both', expand=True)
        v_scrollbar.config(command=text.yview)
        h_scrollbar.config(command=text.xview)

        # Add instructions
        text.insert('1.0', """
BISECTION METHOD

The bisection method finds a root by repeatedly dividing an interval in half.

HOW IT WORKS:
1. Start with an interval [a, b] where f(a) and f(b) have opposite signs
2. Calculate the midpoint c = (a + b) / 2
3. Evaluate f(c)
4. If f(c) is close enough to zero, c is the root
5. Otherwise, replace either [a, c] or [c, b] with the subinterval where the sign change occurs
6. Repeat until the root is found with desired accuracy

PARAMETERS:
• Function f(x): The equation to find roots for
• Lower bound (Xl): Left endpoint of the interval
• Upper bound (Xu): Right endpoint of the interval
• Tolerance: How close to zero f(x) must be to be considered a root
• Error Tolerance (%): Maximum relative error between iterations
• Max Iterations: Maximum number of iterations

WHEN TO USE:
• When you know an interval containing exactly one root
• When you need guaranteed convergence
• When the function is continuous

EXAMPLES:
• x^2 - 2 (root at x = √2 ≈ 1.414)
• e^(-x) - x (root at x ≈ 0.567)
• cos(x) (root at x = π/2)

TIPS:
• The function MUST change sign in the interval
• Check f(Xl) and f(Xu) have opposite signs
• For exponential functions like e^(-x), use parentheses: e^(-x)
• The method always converges but can be slow
• The table is color-coded: green for 1st subinterval, blue for 2nd subinterval
""")
        text.config(state='disabled')  # Make read-only

    def create_newton_raphson_instructions_tab(self):
        tab = ttk.Frame(self.instructions_notebook)
        self.instructions_notebook.add(tab, text="Newton")

        # Create a text widget with scrollbar
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill='both', expand=True, padx=2, pady=2)

        # Add both vertical and horizontal scrollbars
        v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal")
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')

        text = tk.Text(text_frame, width=30, height=15, wrap='none',
                       yscrollcommand=v_scrollbar.set,
                       xscrollcommand=h_scrollbar.set,
                       font=('Arial', 8), bg='#f5f5f5')
        text.pack(side='left', fill='both', expand=True)
        v_scrollbar.config(command=text.yview)
        h_scrollbar.config(command=text.xview)

        # Add instructions
        text.insert('1.0', """
NEWTON-RAPHSON METHOD

The Newton-Raphson method uses tangent lines to quickly converge to a root.

HOW IT WORKS:
1. Start with an initial guess x₀
2. Calculate the tangent line to the function at x₀
3. Find where this tangent line crosses the x-axis (x₁)
4. Repeat the process with x₁ as the new guess
5. Continue until convergence

FORMULA:
x_{i+1} = x_i - f(x_i)/f'(x_i)

PARAMETERS:
• Function f(x): The equation to find roots for
• Initial guess (x₀): Starting point for the iterations
• Tolerance: How close to zero f(x) must be to be considered a root
• Error Tolerance (%): Maximum relative error between iterations
• Max Iterations: Maximum number of iterations

WHEN TO USE:
• When you have a good initial guess
• When the function is smooth and well-behaved
• When you need fast convergence

ADVANTAGES:
• Converges very quickly (quadratic convergence)
• Only needs one initial guess
• Typically requires fewer iterations than bisection

DRAWBACKS:
• May diverge near inflection points
• Division by zero can occur if f'(x) = 0
• May jump to a different root than intended
• Requires calculation of the derivative

EXAMPLES:
• x^2 - 2 (root at x = √2 ≈ 1.414)
• e^(-x) - x (root at x ≈ 0.567)
• 3x + sin(x) - e^x (root at x ≈ -0.36)
""")
        text.config(state='disabled')  # Make read-only

    def create_regula_falsi_instructions_tab(self):
        tab = ttk.Frame(self.instructions_notebook)
        self.instructions_notebook.add(tab, text="Regula Falsi")

        # Create a text widget with scrollbar
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill='both', expand=True, padx=2, pady=2)

        # Add both vertical and horizontal scrollbars
        v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal")
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')

        text = tk.Text(text_frame, width=30, height=15, wrap='none',
                       yscrollcommand=v_scrollbar.set,
                       xscrollcommand=h_scrollbar.set,
                       font=('Arial', 8), bg='#f5f5f5')
        text.pack(side='left', fill='both', expand=True)
        v_scrollbar.config(command=text.yview)
        h_scrollbar.config(command=text.xview)

        # Add instructions
        text.insert('1.0', """
REGULA FALSI METHOD (FALSE POSITION)

The Regula Falsi method combines elements of bisection and secant methods.

HOW IT WORKS:
1. Start with an interval [a, b] where f(a) and f(b) have opposite signs
2. Draw a straight line connecting (a, f(a)) and (b, f(b))
3. Find where this line crosses the x-axis (c)
4. Replace either a or b with c, maintaining opposite signs
5. Repeat until convergence

FORMULA:
x_R = (x_U·f(x_L) - x_L·f(x_U)) / (f(x_L) - f(x_U))

PARAMETERS:
• Function f(x): The equation to find roots for
• Lower bound (XL): Left endpoint of the interval
• Upper bound (XU): Right endpoint of the interval
• Tolerance: How close to zero f(x) must be to be considered a root
• Error Tolerance (%): Maximum relative error between iterations
• Max Iterations: Maximum number of iterations

WHEN TO USE:
• When you need guaranteed convergence like bisection
• When you want faster convergence than bisection
• When the function is continuous

ADVANTAGES:
• Always converges if initial interval contains a root
• Generally faster than bisection method
• Does not require derivative calculation
• More efficient than bisection for asymmetric functions

DRAWBACKS:
• Can be slow for certain functions
• One endpoint may remain fixed for many iterations
• Less efficient than Newton-Raphson for well-behaved functions

EXAMPLES:
• x^3 - 2*x - 5 (root at x ≈ 2.0946)
• e^(-x) - x (root at x ≈ 0.567)
• 3x + sin(x) - e^x (root at x ≈ 0.36)

TIPS:
• The function MUST change sign in the interval
• Check f(XL) and f(XU) have opposite signs
""")
        text.config(state='disabled')  # Make read-only

    def create_secant_instructions_tab(self):
        tab = ttk.Frame(self.instructions_notebook)
        self.instructions_notebook.add(tab, text="Secant")

        # Create a text widget with scrollbar
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill='both', expand=True, padx=2, pady=2)

        # Add both vertical and horizontal scrollbars
        v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal")
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')

        text = tk.Text(text_frame, width=30, height=15, wrap='none',
                       yscrollcommand=v_scrollbar.set,
                       xscrollcommand=h_scrollbar.set,
                       font=('Arial', 8), bg='#f5f5f5')
        text.pack(side='left', fill='both', expand=True)
        v_scrollbar.config(command=text.yview)
        h_scrollbar.config(command=text.xview)

        # Add instructions
        text.insert('1.0', """
SECANT METHOD

The secant method approximates the derivative using two points instead of calculating it directly.

HOW IT WORKS:
1. Start with two initial guesses x₀ and x₁
2. Draw a secant line through (x₀, f(x₀)) and (x₁, f(x₁))
3. Find where this line crosses the x-axis (x₂)
4. Repeat the process with x₁ and x₂ as the new points
5. Continue until convergence

FORMULA:
x_{i+1} = x_i - f(x_i)·(x_i - x_{i-1})/(f(x_i) - f(x_{i-1}))

PARAMETERS:
• Function f(x): The equation to find roots for
• First initial guess (x₀): First point for the secant line
• Second initial guess (x₁): Second point for the secant line
• Tolerance: How close to zero f(x) must be to be considered a root
• Error Tolerance (%): Maximum relative error between iterations
• Max Iterations: Maximum number of iterations

WHEN TO USE:
• When calculating derivatives is difficult or expensive
• When you have two good initial guesses
• When Newton-Raphson method fails or is impractical

ADVANTAGES:
• Faster convergence than bisection and false position
• Does not require derivative calculation
• Often converges almost as fast as Newton-Raphson
• More efficient than Newton-Raphson for complex functions

DRAWBACKS:
• Requires two initial guesses
• No guarantee of convergence
• Can fail if the secant becomes nearly horizontal
• May diverge for certain functions

EXAMPLES:
• x^2 - 2 (root at x = √2 ≈ 1.414)
• e^(-x) - x (root at x ≈ 0.567)
• 3x + sin(x) - e^x (root at x ≈ -0.36)
""")
        text.config(state='disabled')  # Make read-only

    def create_legends_tab(self):
        tab = ttk.Frame(self.instructions_notebook)
        self.instructions_notebook.add(tab, text="Legends")

        # Create a text widget with scrollbar
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill='both', expand=True, padx=2, pady=2)

        # Add both vertical and horizontal scrollbars
        v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal")
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')

        text = tk.Text(text_frame, width=30, height=15, wrap='none',
                       yscrollcommand=v_scrollbar.set,
                       xscrollcommand=h_scrollbar.set,
                       font=('Arial', 8), bg='#f5f5f5')
        text.pack(side='left', fill='both', expand=True)
        v_scrollbar.config(command=text.yview)
        h_scrollbar.config(command=text.xview)

        # Add legends
        text.insert('1.0', """
LEGENDS AND SYMBOLS

PLOT ELEMENTS:
• Blue line: The function f(x)
• Black horizontal line: The x-axis (y = 0)
• Orange dots: Evaluation points
• Green vertical dashed line: Location of a root
• Green dot: Root (where function crosses x-axis)
• Root label: Shows the exact value of the root

TABLE COLUMNS (INCREMENTAL SEARCH):
• Iteration: Step number
• Xl: Current x value
• Δx: Step size
• Xu: Next x value (Xl + Δx)
• f(Xl): Function value at Xl
• f(Xu): Function value at Xu
• Product: Sign of f(Xl) × f(Xu)
• Remark: Status of the search

TABLE COLUMNS (BISECTION METHOD):
• Iteration: Step number
• Xl: Lower bound of current interval
• Xr: Midpoint of current interval
• Xu: Upper bound of current interval
• f(Xl): Function value at Xl
• f(Xr): Function value at Xr
• |Ea|,%: Absolute relative approximate error
• f(Xl).f(Xr): Sign of f(Xl) × f(Xr)
• Remark: Which subinterval contains the root

TABLE COLUMNS (NEWTON-RAPHSON):
• Iteration: Step number
• Xi: Current x value
• |Ea|,%: Absolute relative approximate error
• f(Xi): Function value at Xi
• f'(Xi): Derivative value at Xi
• Remark: Status of the iteration

TABLE COLUMNS (REGULA FALSI):
• Iteration: Step number
• XL: Lower bound of current interval
• XU: Upper bound of current interval
• XR: False position point
• |Ea|,%: Absolute relative approximate error
• f(XL): Function value at XL
• f(XU): Function value at XU
• f(XR): Function value at XR
• f(XL)*f(XR): Sign of f(XL) × f(XR)
• Remark: Which endpoint was replaced

TABLE COLUMNS (SECANT METHOD):
• Iteration: Step number
• Xi-1: Previous x value
• Xi: Current x value
• Xi+1: Next x value
• |Ea|,%: Absolute relative approximate error
• f(Xi-1): Function value at Xi-1
• f(Xi): Function value at Xi
• f(Xi+1): Function value at Xi+1
• Remark: Status of the iteration

COLOR CODING (BISECTION TABLE):
• Green rows: Root is in the 1st subinterval [Xl, Xr]
• Blue rows: Root is in the 2nd subinterval [Xr, Xu]

SPECIAL FUNCTIONS:
• e or exp(x): Exponential function (e^x)
• ln(x): Natural logarithm (base e)
• log(x): Base-10 logarithm
• sqrt(x): Square root
• sin(x), cos(x), tan(x): Trigonometric functions
""")
        text.config(state='disabled')  # Make read-only

    def enable_notebook_scrolling(self, notebook):
        """Enable horizontal scrolling for notebook tabs"""

        # Create a frame to hold the notebook
        frame = ttk.Frame(notebook.master)
        frame.pack(fill='both', expand=True)

        # Create left and right scroll buttons
        left_button = ttk.Button(frame, text="◄", width=2,
                                 command=lambda: self.scroll_tabs(notebook, -1),
                                 style='Menu.TButton')
        right_button = ttk.Button(frame, text="►", width=2,
                                  command=lambda: self.scroll_tabs(notebook, 1),
                                  style='Menu.TButton')

        # Place the buttons and notebook
        left_button.pack(side='left', fill='y')
        right_button.pack(side='right', fill='y')
        notebook.pack(side='left', fill='both', expand=True)

        # Store the current tab index
        notebook.current_tab = 0

        # Bind mouse wheel to scroll tabs
        notebook.bind("<MouseWheel>", lambda event: self.scroll_tabs(notebook, -1 if event.delta > 0 else 1))
        notebook.bind("<Button-4>", lambda event: self.scroll_tabs(notebook, -1))  # Linux scroll up
        notebook.bind("<Button-5>", lambda event: self.scroll_tabs(notebook, 1))  # Linux scroll down

    def scroll_tabs(self, notebook, direction):
        """Scroll the notebook tabs in the given direction"""
        if not notebook.tabs():
            return

        current = notebook.index(notebook.select())
        max_tabs = len(notebook.tabs()) - 1

        # Calculate the new tab index
        new_index = current + direction
        if new_index < 0:
            new_index = 0
        elif new_index > max_tabs:
            new_index = max_tabs

        # Select the new tab
        if new_index != current:
            notebook.select(new_index)

    def show_instructions(self):
        # Make sure the instructions tab is visible
        self.instructions_notebook.select(0)  # Select the General tab
        messagebox.showinfo("Instructions", "Instructions are available in the tabs at the bottom of the sidebar.")

    def create_graphical_frame(self):
        frame = ttk.Frame(self.content_area)
        # This frame is now empty since parameters are in the top input panel
        return frame

    def create_incremental_frame(self):
        frame = ttk.Frame(self.content_area)
        # This frame is now empty since parameters are in the top input panel
        return frame

    def create_bisection_frame(self):
        frame = ttk.Frame(self.content_area)
        # This frame is now empty since parameters are in the top input panel
        return frame

    def create_newton_raphson_frame(self):
        frame = ttk.Frame(self.content_area)
        # This frame is now empty since parameters are in the top input panel
        return frame

    def create_regula_falsi_frame(self):
        frame = ttk.Frame(self.content_area)
        # This frame is now empty since parameters are in the top input panel
        return frame

    def create_secant_frame(self):
        frame = ttk.Frame(self.content_area)
        # This frame is now empty since parameters are in the top input panel
        return frame

    def preview_equation(self, equation):
        """Preview how the equation will be interpreted"""
        if equation:
            try:
                processed = preprocess_equation(equation)
                self.visuals.update_equation_label(equation)
                messagebox.showinfo("Equation Preview", f"Your equation will be interpreted as:\n\n{processed}")
            except Exception as e:
                messagebox.showerror("Parsing Error", f"Error parsing equation: {str(e)}")

    def show_method(self, method_key):
        # Hide all parameter frames
        for frame in self.param_frames.values():
            frame.grid_forget()

        # Show selected parameter frame
        self.param_frames[method_key].grid(row=1, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        # Update the common function entry with the method-specific function
        if method_key == "graphical" and self.graph_function_entry:
            self.common_function_entry.delete(0, tk.END)
            self.common_function_entry.insert(0, self.graph_function_entry.get())
        elif method_key == "incremental" and self.incr_function_entry:
            self.common_function_entry.delete(0, tk.END)
            self.common_function_entry.insert(0, self.incr_function_entry.get())
        elif method_key == "bisection" and self.bisection_function_entry:
            self.common_function_entry.delete(0, tk.END)
            self.common_function_entry.insert(0, self.bisection_function_entry.get())
        elif method_key == "newton_raphson" and self.newton_function_entry:
            self.common_function_entry.delete(0, tk.END)
            self.common_function_entry.insert(0, self.newton_function_entry.get())
        elif method_key == "regula_falsi" and self.regula_falsi_function_entry:
            self.common_function_entry.delete(0, tk.END)
            self.common_function_entry.insert(0, self.regula_falsi_function_entry.get())
        elif method_key == "secant" and self.secant_function_entry:
            self.common_function_entry.delete(0, tk.END)
            self.common_function_entry.insert(0, self.secant_function_entry.get())

        # Clear visuals
        self.visuals.clear_all()

        # Update method indicator in header
        method_name = method_key.replace('_', '-').title()
        self.method_indicator.config(text=f"{method_name} Method Selected")

        # Update table view based on method
        self.visuals.show_table(method_key)

        # Select the corresponding instructions tab
        if method_key == "graphical":
            self.instructions_notebook.select(1)  # Graphical tab
        elif method_key == "incremental":
            self.instructions_notebook.select(2)  # Incremental tab
        elif method_key == "bisection":
            self.instructions_notebook.select(3)  # Bisection tab
        elif method_key == "newton_raphson":
            self.instructions_notebook.select(4)  # Newton-Raphson tab
        elif method_key == "regula_falsi":
            self.instructions_notebook.select(5)  # Regula Falsi tab
        elif method_key == "secant":
            self.instructions_notebook.select(6)  # Secant tab

        # Highlight the active method button
        for key, button in self.method_buttons.items():
            if key == method_key:
                button.configure(style='ActiveMenu.TButton')
            else:
                button.configure(style='Menu.TButton')














    def compute_incremental_search(self):
        # Get the equation from the common input field
        eqn = self.common_function_entry.get()
        # Update the method-specific entry
        self.incr_function_entry.delete(0, tk.END)
        self.incr_function_entry.insert(0, eqn)

        # Get other parameters
        x_start = self.incr_start_entry.get()
        step = self.incr_step_entry.get()
        tolerance = self.incr_tolerance_entry.get()
        max_iter = self.incr_max_iter_entry.get()

        try:
            if not eqn:
                raise ValueError("Function cannot be empty")

            x_start = float(x_start)
            step = float(step)
            tolerance = float(tolerance)
            max_iter = int(max_iter)

            # Update equation interpretation label
            self.visuals.update_equation_label(eqn)

            roots, table_data, points = incremental_search(
                eqn=eqn,
                x_start=x_start,
                initial_step=step,
                tolerance=tolerance,
                max_iterations=max_iter
            )

            # Format roots output
            if not roots:
                self.visuals.update_result_label("No roots found in the specified range")
            else:
                if len(roots) == 1:
                    root_text = f"Root found at x ≈ {roots[0]:.6f}"
                else:
                    root_text = f"Found {len(roots)} roots:\n" + "\n".join([f"• x ≈ {root:.6f}" for root in roots])

                self.visuals.update_result_label(root_text)

            self.visuals.update_plot(points, eqn, roots)
            self.visuals.update_table(table_data)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"An error occurred: {str(e)}")








    def compute_graphical_method(self):
        # Get the equation from the common input field
        eqn = self.common_function_entry.get()
        # Update the method-specific entry
        self.graph_function_entry.delete(0, tk.END)
        self.graph_function_entry.insert(0, eqn)

        x_lower = self.graph_lower_entry.get()
        x_upper = self.graph_upper_entry.get()

        try:
            if not eqn:
                raise ValueError("Function cannot be empty")

            x_lower = float(x_lower)
            x_upper = float(x_upper)

            if x_lower >= x_upper:
                raise ValueError("Lower bound must be less than upper bound")

            # Update equation interpretation label
            self.visuals.update_equation_label(eqn)

            roots, points = graphical_method(eqn, x_lower, x_upper)

            if not roots:
                self.visuals.update_result_label("No roots detected visually in this range")
            else:
                if len(roots) == 1:
                    root_text = f"Graphical root estimated at x ≈ {roots[0]:.6f}"
                else:
                    root_text = f"Found {len(roots)} graphical roots:\n" + "\n".join(
                        [f"• x ≈ {root:.6f}" for root in roots])
                self.visuals.update_result_label(root_text)

            self.visuals.update_plot(points, eqn, roots if roots else None)
            self.visuals.update_table([])

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Graphical Method Error", f"An error occurred: {str(e)}")










    def compute_bisection_method(self):
        # Get the equation from the common input field
        eqn = self.common_function_entry.get()
        # Update the method-specific entry
        self.bisection_function_entry.delete(0, tk.END)
        self.bisection_function_entry.insert(0, eqn)

        x_lower = self.bisection_lower_entry.get()
        x_upper = self.bisection_upper_entry.get()
        tolerance = self.bisection_tolerance_entry.get()
        error_tolerance = self.bisection_error_entry.get()
        max_iter = self.bisection_max_iter_entry.get()

        try:
            if not eqn:
                raise ValueError("Function cannot be empty")

            x_lower = float(x_lower)
            x_upper = float(x_upper)
            tolerance = float(tolerance)
            error_tolerance = float(error_tolerance)
            max_iter = int(max_iter)

            if x_lower >= x_upper:
                raise ValueError("Lower bound must be less than upper bound")

            # Update equation interpretation label
            self.visuals.update_equation_label(eqn)

            # Call the updated bisection method that finds multiple roots
            roots, table_data, points = bisection_method(
                eqn=eqn,
                x_lower=x_lower,
                x_upper=x_upper,
                tolerance=tolerance,
                max_iterations=max_iter,
                error_tolerance=error_tolerance,
                find_multiple_roots=True  # Always find multiple roots
            )

            # Format roots output
            if not roots:
                self.visuals.update_result_label(f"No roots found in the interval [{x_lower}, {x_upper}].")
            else:
                if len(roots) == 1:
                    root_text = f"Root found at x ≈ {roots[0]:.6f}"
                else:
                    root_text = f"Found {len(roots)} roots:\n" + "\n".join([f"• x ≈ {root:.6f}" for root in roots])
                self.visuals.update_result_label(root_text)

            self.visuals.update_plot(points, eqn, roots)
            self.visuals.update_table(table_data)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Bisection Method Error", f"An error occurred: {str(e)}")









    def compute_newton_raphson_method(self):
        # Get the equation from the common input field
        eqn = self.common_function_entry.get()
        # Update the method-specific entry
        self.newton_function_entry.delete(0, tk.END)
        self.newton_function_entry.insert(0, eqn)

        x_initial = self.newton_initial_entry.get()
        tolerance = self.newton_tolerance_entry.get()
        error_tolerance = self.newton_error_entry.get()
        max_iter = self.newton_max_iter_entry.get()

        try:
            if not eqn:
                raise ValueError("Function cannot be empty")

            x_initial = float(x_initial)
            tolerance = float(tolerance)
            error_tolerance = float(error_tolerance)
            max_iter = int(max_iter)

            # Update equation interpretation label
            self.visuals.update_equation_label(eqn)

            # Use search range of -10 to 10 for finding multiple roots
            search_range = (-10, 10)

            roots, table_data, points = newton_raphson_method(
                eqn=eqn,
                x_initial=x_initial,
                tolerance=tolerance,
                max_iterations=max_iter,
                error_tolerance=error_tolerance,
                find_multiple_roots=True,
                search_range=search_range
            )

            # Format roots output
            if not roots:
                self.visuals.update_result_label(f"Newton-Raphson method failed to find any roots.")
            else:
                if isinstance(roots, list):
                    if len(roots) == 1:
                        root_text = f"Root found at x ≈ {roots[0]:.6f}"
                    else:
                        root_text = f"Found {len(roots)} roots:\n" + "\n".join([f"• x ≈ {root:.6f}" for root in roots])
                else:
                    # Handle case where a single root is returned (not in a list)
                    root_text = f"Root found at x ≈ {roots:.6f}"
                    roots = [roots]  # Convert to list for plotting

                self.visuals.update_result_label(root_text)

            self.visuals.update_plot(points, eqn, roots)
            self.visuals.update_table(table_data)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Newton-Raphson Method Error", f"An error occurred: {str(e)}")













    def compute_regula_falsi_method(self):
        # Get the equation from the common input field
        eqn = self.common_function_entry.get()
        # Update the method-specific entry
        self.regula_falsi_function_entry.delete(0, tk.END)
        self.regula_falsi_function_entry.insert(0, eqn)

        x_lower = self.regula_falsi_lower_entry.get()
        x_upper = self.regula_falsi_upper_entry.get()
        tolerance = self.regula_falsi_tolerance_entry.get()
        error_tolerance = self.regula_falsi_error_entry.get()
        max_iter = self.regula_falsi_max_iter_entry.get()

        try:
            if not eqn:
                raise ValueError("Function cannot be empty")

            x_lower = float(x_lower)
            x_upper = float(x_upper)
            tolerance = float(tolerance)
            error_tolerance = float(error_tolerance)
            max_iter = int(max_iter)

            if x_lower >= x_upper:
                raise ValueError("Lower bound must be less than upper bound")

            # Update equation interpretation label
            self.visuals.update_equation_label(eqn)

            # Check function values at endpoints
            f_lower = evaluate_function(x_lower, eqn)
            f_upper = evaluate_function(x_upper, eqn)

            if f_lower * f_upper > 0:
                messagebox.showwarning("No Sign Change",
                                       f"Function does not change sign in the interval [{x_lower}, {x_upper}]:\n"
                                       f"f({x_lower}) = {f_lower:.6f}\n"
                                       f"f({x_upper}) = {f_upper:.6f}\n\n"
                                       f"The Regula Falsi method works best with a sign change.")

            roots, table_data, points = regula_falsi_method(
                eqn=eqn,
                x_lower=x_lower,
                x_upper=x_upper,
                tolerance=tolerance,
                max_iterations=max_iter,
                error_tolerance=error_tolerance,
                allow_no_sign_change=True,
                find_multiple_roots=True
            )

            # Format roots output
            if not roots:
                self.visuals.update_result_label(f"No roots found in the interval [{x_lower}, {x_upper}].\n"
                                                 f"f({x_lower}) = {f_lower:.6f}, f({x_upper}) = {f_upper:.6f}")
            else:
                if isinstance(roots, list):
                    if len(roots) == 1:
                        root_text = f"Root found at x ≈ {roots[0]:.6f}"
                    else:
                        root_text = f"Found {len(roots)} roots:\n" + "\n".join([f"• x ≈ {root:.6f}" for root in roots])
                else:
                    # Handle case where a single root is returned (not in a list)
                    root_text = f"Root found at x ≈ {roots:.6f}"
                    roots = [roots]  # Convert to list for plotting

                self.visuals.update_result_label(root_text)

            self.visuals.update_plot(points, eqn, roots)
            self.visuals.update_table(table_data)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Regula Falsi Method Error", f"An error occurred: {str(e)}")













    def compute_secant_method(self):
        # Get the equation from the common input field
        eqn = self.common_function_entry.get()
        # Update the method-specific entry
        self.secant_function_entry.delete(0, tk.END)
        self.secant_function_entry.insert(0, eqn)

        x0 = self.secant_x0_entry.get()
        x1 = self.secant_x1_entry.get()
        tolerance = self.secant_tolerance_entry.get()
        error_tolerance = self.secant_error_entry.get()
        max_iter = self.secant_max_iter_entry.get()

        try:
            if not eqn:
                raise ValueError("Function cannot be empty")

            x0 = float(x0)
            x1 = float(x1)
            tolerance = float(tolerance)
            error_tolerance = float(error_tolerance)
            max_iter = int(max_iter)

            if x0 == x1:
                raise ValueError("First and second guesses must be different")

            # Update equation interpretation label
            self.visuals.update_equation_label(eqn)

            # Use search range of -10 to 10 for finding multiple roots
            search_range = (-10, 10)

            roots, table_data, points = secant_method(
                eqn=eqn,
                x0=x0,
                x1=x1,
                tolerance=tolerance,
                max_iterations=max_iter,
                error_tolerance=error_tolerance,
                find_multiple_roots=True,
                search_range=search_range
            )

            # Format roots output
            if not roots:
                self.visuals.update_result_label(f"Secant method failed to find any roots.")
            else:
                if isinstance(roots, list):
                    if len(roots) == 1:
                        root_text = f"Root found at x ≈ {roots[0]:.6f}"
                    else:
                        root_text = f"Found {len(roots)} roots:\n" + "\n".join([f"• x ≈ {root:.6f}" for root in roots])
                else:
                    # Handle case where a single root is returned (not in a list)
                    root_text = f"Root found at x ≈ {roots:.6f}"
                    roots = [roots]  # Convert to list for plotting

                self.visuals.update_result_label(root_text)

            self.visuals.update_plot(points, eqn, roots)
            self.visuals.update_table(table_data)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Secant Method Error", f"An error occurred: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = RootFinderApp(root)
    root.mainloop()