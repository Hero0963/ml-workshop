<script>
  import { onMount, tick } from "svelte";

  const API_BASE_URL = "http://127.0.0.1:7440";

  // --- State Variables ---
  let rows = 6;
  let cols = 6;
  let puzzle_grid = []; // 2D array for cell values
  let walls = new Set(); // Set to store wall tuples as JSON strings
  
  let solver = "DFS";
  let solvers = ["DFS", "A* (heapq)", "CP-SAT"];

  let solution_path = "";
  let solution_gif = null;
  let solution_image = null;
  let loading = false;
  let error_message = "";

  // --- Canvas and Drawing ---
  let canvas;
  let ctx;
  const cell_size = 50;
  const margin = 10;
  let canvas_width = 0;
  let canvas_height = 0;

  // --- In-place Editing State ---
  let editing_cell = null; // e.g., { r, c }
  let input_element; // To bind to the <input>

  // --- Reactive Statements ---
  $: if (rows > 0 && cols > 0 && typeof window !== 'undefined') {
    create_grid();
  }
  $: {
    canvas_width = cols * cell_size + 2 * margin;
    canvas_height = rows * cell_size + 2 * margin;
  }
  $: if (ctx && puzzle_grid && walls) {
    draw_puzzle();
  }
  // When editing_cell changes, focus the new input element
  $: if (editing_cell && input_element) {
      input_element.value = puzzle_grid[editing_cell.r][editing_cell.c] || "";
      input_element.focus();
      input_element.select();
  }

  onMount(() => {
    ctx = canvas.getContext("2d");
  });

  // --- Core Functions ---

  function create_grid() {
    let new_grid = [];
    for (let r = 0; r < rows; r++) {
      new_grid.push(new Array(cols).fill(""));
    }
    puzzle_grid = new_grid;
  }

  function reset_app() {
      rows = 6;
      cols = 6;
      create_grid();
      walls = new Set(); // Re-assign for reactivity
      reset_solution();
  }

  function reset_solution() {
      solution_path = "";
      solution_gif = null;
      solution_image = null;
      error_message = "";
  }

  async function solve_puzzle() {
    loading = true;
    reset_solution();

    const puzzle_list = puzzle_grid.map(row => 
        row.map(cell => {
            const val = String(cell).trim().toLowerCase();
            if (val === 'x') return 'xx';
            if (val === '') return '  ';
            return String(cell).trim();
        })
    );
    const walls_list = Array.from(walls).map(JSON.parse);

    let walls_str;
    if (walls_list.length === 0) {
        walls_str = "set()"; // Python empty set literal
    } else {
        const wall_tuples_str = walls_list.map(wall => {
            const [[r1, c1], [r2, c2]] = wall;
            const p1 = `(${r1},${c1})`;
            const p2 = `(${r2},${c2})`;
            return `(${p1},${p2})`;
        }).join(", ");
        walls_str = `{${wall_tuples_str}}`;
    }

    const payload = {
        puzzle_layout_str: JSON.stringify(puzzle_list),
        walls_str: walls_str, // Use the manually constructed Python set string
        solver_name: solver
    };

    try {
      const response = await fetch(`${API_BASE_URL}/api/solver/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
          const err_data = await response.json();
          throw new Error(err_data.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      solution_path = data.solution_path || "";
      if (data.solution_gif_b64) {
          solution_gif = `data:image/gif;base64,${data.solution_gif_b64}`;
      }
      if (data.solution_final_image_b64) {
          solution_image = `data:image/png;base64,${data.solution_final_image_b64}`;
      }

    } catch (e) {
        error_message = e.message;
    } finally {
        loading = false;
    }
  }

  // --- Drawing Functions ---
  function draw_puzzle() {
    if (!ctx || !puzzle_grid || puzzle_grid.length === 0) return;

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas_width, canvas_height);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = margin + c * cell_size;
        const y = margin + r * cell_size;
        const val = puzzle_grid[r] ? puzzle_grid[r][c] : '';
        const is_editing = editing_cell && editing_cell.r === r && editing_cell.c === c;

        if (String(val).toLowerCase() === 'x') {
            ctx.fillStyle = "black";
            ctx.fillRect(x, y, cell_size, cell_size);
        } else {
            ctx.strokeStyle = "#ccc";
            ctx.strokeRect(x, y, cell_size, cell_size);
            if (val && !is_editing) {
                ctx.fillStyle = "black";
                ctx.font = `${cell_size * 0.4}px Arial`;
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(String(val), x + cell_size / 2, y + cell_size / 2);
            }
        }
      }
    }

    ctx.strokeStyle = "red";
    ctx.lineWidth = 3;
    for (const wall_str of walls) {
        const wall = JSON.parse(wall_str);
        const [[r1, c1], [r2, c2]] = wall;
        if (r1 === r2) { // Vertical wall
            const line_x = margin + Math.max(c1, c2) * cell_size;
            const line_y0 = margin + r1 * cell_size;
            ctx.beginPath();
            ctx.moveTo(line_x, line_y0);
            ctx.lineTo(line_x, line_y0 + cell_size);
            ctx.stroke();
        } else { // Horizontal wall
            const line_x0 = margin + c1 * cell_size;
            const line_y = margin + Math.max(r1, r2) * cell_size;
            ctx.beginPath();
            ctx.moveTo(line_x0, line_y);
            ctx.lineTo(line_x0 + cell_size, line_y);
            ctx.stroke();
        }
    }
    ctx.lineWidth = 1;
  }

  // --- INTERACTION HANDLER ---
  function handle_canvas_click(event) {
    // If we are already editing, a click outside the input should save and hide it.
    if (editing_cell && event.target !== input_element) {
        handle_input_blur();
    }

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    if (x < margin || x > canvas_width - margin || y < margin || y > canvas_height - margin) return;

    const c = Math.floor((x - margin) / cell_size);
    const r = Math.floor((y - margin) / cell_size);

    const x_in_cell = (x - margin) % cell_size;
    const y_in_cell = (y - margin) % cell_size;
    const border_threshold = 8; // 8px threshold for clicking a border

    let wall_operated = false;

    // Check for border clicks
    if (x_in_cell < border_threshold && c > 0) { // Left border
        toggle_wall([r, c-1], [r, c]);
        wall_operated = true;
    } else if (x_in_cell > cell_size - border_threshold && c < cols - 1) { // Right border
        toggle_wall([r, c], [r, c+1]);
        wall_operated = true;
    } else if (y_in_cell < border_threshold && r > 0) { // Top border
        toggle_wall([r-1, c], [r, c]);
        wall_operated = true;
    } else if (y_in_cell > cell_size - border_threshold && r < rows - 1) { // Bottom border
        toggle_wall([r, c], [r+1, c]);
        wall_operated = true;
    }

    // If no border was clicked, it's a cell click
    if (!wall_operated) {
        editing_cell = { r, c };
    }
  }

  function toggle_wall(c1, c2) {
      const wall = JSON.stringify(c1 > c2 ? [c2, c1] : [c1, c2]);
      const new_walls = new Set(walls);
      if (new_walls.has(wall)) {
          new_walls.delete(wall);
      } else {
          new_walls.add(wall);
      }
      walls = new_walls; // Re-assign to trigger reactivity
  }

  function handle_input_blur() {
      if (!editing_cell) return;
      
      const { r, c } = editing_cell;
      const new_value = input_element.value;

      const new_grid = [...puzzle_grid];
      new_grid[r][c] = new_value;
      puzzle_grid = new_grid;

      editing_cell = null; // Hide the input
      draw_puzzle(); // Redraw to show the new value
  }

  function handle_input_keydown(event) {
      if (event.key === 'Enter') {
          input_element.blur(); // Triggers the blur handler
      } else if (event.key === 'Escape') {
          editing_cell = null; // Cancel editing, no change
          draw_puzzle(); // Redraw to show original value
      }
  }

</script>

<main>
  <h1>Zip Puzzle Editor & Solver</h1>
  <div class="container">
    <!-- CONTROLS -->
    <div class="editor-pane">
      <div class="control-group">
        <h2>1. Controls</h2>
        <label>Rows: <input type="number" bind:value={rows} min=1 class="control-input"></label>
        <label>Cols: <input type="number" bind:value={cols} min=1 class="control-input"></label>
        <br>
        <button on:click={reset_app}>Reset Puzzle</button>
      </div>

      <div class="control-group">
        <h2>2. Solve</h2>
        <label>Solver: 
            <select bind:value={solver}>
                {#each solvers as s}
                    <option value={s}>{s}</option>
                {/each}
            </select>
        </label>
        <button on:click={solve_puzzle} disabled={loading}>
            {#if loading}Solving...{:else}Solve Puzzle{/if}
        </button>
      </div>

       <div class="control-group">
        <h2>Instructions</h2>
        <ul>
            <li>Click in the **middle** of a cell to set its value (number or 'x').</li>
            <li>Click on a **border** between cells to toggle a wall.</li>
            <li>Use the controls above to resize or reset the grid.</li>
        </ul>
      </div>

    </div>

    <!-- EDITOR, PREVIEW & SOLUTION -->
    <div class="preview-pane">
        <h2>Puzzle Editor</h2>
        <div class="canvas-wrapper" style="position: relative;">
            <canvas bind:this={canvas} width={canvas_width} height={canvas_height} on:click={handle_canvas_click}></canvas>
            {#if editing_cell}
                <input
                    bind:this={input_element}
                    type="text"
                    class="cell-input"
                    style="left: {margin + editing_cell.c * cell_size}px; top: {margin + editing_cell.r * cell_size}px; width: {cell_size}px; height: {cell_size}px;"
                    on:blur={handle_input_blur}
                    on:keydown={handle_input_keydown}
                />
            {/if}
        </div>
        
        {#if error_message}
            <div class="solution-box error">
                <h3>Error</h3>
                <p>{error_message}</p>
            </div>
        {/if}

        {#if solution_path}
            <div class="solution-box">
                <h3>Solution</h3>
                <p><b>Path:</b> {solution_path}</p>
                <h4>Animation:</h4>
                <img src={solution_gif} alt="Solution Animation" />
                <h4>Final State:</h4>
                <img src={solution_image} alt="Final Solution" />
            </div>
        {/if}
    </div>
  </div>
</main>

<style>
  :root {
      --border-color: #eee;
  }
  main {
    font-family: sans-serif;
  }
  .container {
    display: flex;
    gap: 2rem;
  }
  .editor-pane, .preview-pane {
    flex: 1;
    padding: 1rem;
    border: 1px solid var(--border-color);
    border-radius: 8px;
  }
  .preview-pane {
      flex: 2;
      display: flex;
      flex-direction: column;
      align-items: center;
  }
  .control-group {
      margin-bottom: 2rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border-color);
  }
  .control-input {
      width: 60px;
  }
  .canvas-wrapper {
      position: relative;
      border: 2px solid black;
  }
  canvas {
      cursor: crosshair;
      display: block; /* Removes bottom space under canvas */
  }
  .cell-input {
      position: absolute;
      box-sizing: border-box; /* Include padding and border in the element's total width and height */
      text-align: center;
      font-family: Arial, sans-serif;
      font-size: calc(50px * 0.4); /* Match canvas font size */
      border: 1px solid #007bff;
      padding: 0;
  }
  .solution-box img {
      max-width: 100%;
      border: 1px solid #ddd;
      margin-top: 1rem;
  }
  .error {
      color: red;
      border: 1px solid red;
      padding: 1rem;
      margin-top: 1rem;
  }
</style>