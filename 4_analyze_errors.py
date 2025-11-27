import json
import os
import pandas as pd
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# --- CONFIGURATION ---
JSON_FILE = r"tsne_results\file.json"
K_NEIGHBORS = 5 
CONFLICT_THRESHOLD = 0.6 

def main():
    if not os.path.exists(JSON_FILE):
        print(f"Error: File not found: {JSON_FILE}")
        return

    print(f"Loading data from {JSON_FILE}...")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # --- 1. SORT CLASSES (Fixes the messy legend) ---
    df.sort_values(by=['class', 'filename'], inplace=True)
    all_classes = sorted(df['class'].unique())
    print(f"Processing {len(all_classes)} unique classes...")

    # --- 2. DETECT CONFLICTS ---
    coords = df[['x', 'y']].values
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    status_list = []
    suggestion_list = []
    score_list = []
    opacity_list = []
    size_list = []
    line_width_list = []

    for i in range(len(df)):
        neighbor_indices = indices[i][1:]
        current_class = df.iloc[i]['class']
        neighbor_classes = df.iloc[neighbor_indices]['class'].tolist()
        
        counts = Counter(neighbor_classes)
        most_common_neighbor, _ = counts.most_common(1)[0]
        mismatch_count = sum(1 for c in neighbor_classes if c != current_class)
        conflict_score = mismatch_count / K_NEIGHBORS
        
        if conflict_score >= CONFLICT_THRESHOLD:
            # IT'S AN ERROR (Highlight this!)
            status_list.append("SUSPICIOUS")
            suggestion_list.append(f"Likely: {most_common_neighbor}")
            score_list.append(f"{conflict_score:.2f}")
            
            # VISUAL STYLING FOR ERRORS:
            opacity_list.append(1.0)  # Fully Opaque (Solid)
            size_list.append(12)      # Big
            line_width_list.append(2) # Thick black border
        else:
            # IT'S CLEAN (Make this background noise)
            status_list.append("Clean")
            suggestion_list.append("Correct")
            score_list.append("0.0")
            
            # VISUAL STYLING FOR CLEAN DATA:
            opacity_list.append(0.15) # Very Faint (Ghost)
            size_list.append(5)       # Small
            line_width_list.append(0) # No border

    df['status'] = status_list
    df['suggestion'] = suggestion_list
    df['conflict_score'] = score_list
    df['viz_opacity'] = opacity_list
    df['viz_size'] = size_list
    df['viz_border'] = line_width_list

    sus_count = len(df[df['status'] == "SUSPICIOUS"])
    print(f"Found {sus_count} potential labeling errors.")

    # --- 3. CREATE "FOCUS MODE" PLOT ---
    print("Generating Focus Map...")
    
    # We construct a custom palette to handle many classes better
    # Combining multiple Plotly palettes to get ~50 distinct colors before repeating
    large_palette = (
        px.colors.qualitative.Alphabet + 
        px.colors.qualitative.Dark24 + 
        px.colors.qualitative.Light24
    )

    fig = px.scatter(
        df,
        x='x', 
        y='y',
        color='class',             # Color determines identity
        symbol='status',           # Shape determines error status
        symbol_map={
            "Clean": "circle", 
            "SUSPICIOUS": "diamond" # Diamond stands out against circles
        },
        category_orders={"class": all_classes}, # ENFORCE SORTING
        color_discrete_sequence=large_palette,  # BETTER COLORS
        hover_data=['filename', 'suggestion', 'conflict_score'],
        title=f"Focus Analysis: {sus_count} Errors (Diamonds) vs Context (Faint Circles)",
        width=1400, 
        height=1000,
        template="plotly_white"
    )

    # --- 4. APPLY CUSTOM STYLING TRICKS ---
    # Plotly Express doesn't let us map opacity/size easily per-point via parameters
    # so we update the traces manually to achieve the "Ghost Effect".
    
    fig.update_traces(
        marker=dict(
            size=8, 
            opacity=0.6, # Default fallback
            line=dict(width=0.5, color='DarkSlateGrey')
        )
    )

    # We need to iterate through the figure data to apply specific opacity to specific points
    # Since Plotly splits traces by "Class" + "Symbol", we have many traces.
    # This loop ensures "Clean" traces are faint and "Suspicious" traces are bold.
    for trace in fig.data:
        if "diamond" in trace.marker.symbol: 
            # This is a SUSPICIOUS trace
            trace.marker.opacity = 1.0
            trace.marker.size = 12
            trace.marker.line.width = 2
            trace.marker.line.color = 'black' # High contrast border
        else:
            # This is a CLEAN trace
            trace.marker.opacity = 0.2 # Ghost mode
            trace.marker.size = 5
            trace.marker.line.width = 0

    # Save
    output_html = JSON_FILE.replace(".json", "_focus_map.html")
    fig.write_html(output_html)
    print(f"Map saved: {output_html}")
    
    # Export CSV for workflow
    if sus_count > 0:
        sus_df = df[df['status'] == "SUSPICIOUS"]
        csv_path = JSON_FILE.replace(".json", "_errors_sorted.csv")
        # Sort by suggested label so you can fix all "Cats labeled as Dogs" at once
        sus_df = sus_df.sort_values(by=['suggestion', 'conflict_score'], ascending=[True, False])
        sus_df[['filename', 'class', 'suggestion', 'conflict_score', 'path']].to_csv(csv_path, index=False)
        print(f"Prioritized fix list saved: {csv_path}")

    import webbrowser
    webbrowser.open(os.path.abspath(output_html))

if __name__ == "__main__":
    main()