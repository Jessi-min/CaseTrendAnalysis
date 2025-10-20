# -*- coding: utf-8 -*-
"""
Dynamic FPS + Camera logcat plotter (Bokeh)

Usage:
  python plot_dynamic_fps_bokeh.py -i DynamicFPS_logcat.txt -o dynamic_fps_plot.html

What it plots:
- Measured fps (left axis, line + scatter)
- measuredUtil (right axis, line + scatter)
- fps control threshold (left axis, dashed step)
- HintFps & TargetFps (left axis, step + optional markers; values forward-filled)
- Camera series from collectAndDecide:
    * Camera FPS (left axis, line + scatter)
    * Camera UTIL (right axis, line + scatter)
    * Applied Profile (step, on a discrete camera axis)
- Camera event/state from processCameraEvent (both as step lines on camera axis)
Timeline highlights:
- readAndApplyProfile / releaseProfileLock / Touch Event Received
- sendEvents → HWDecode hint / HWDecode default hint
- sendEvents → Camera preview hint / Camera close hint

Tooltips:
- Per-series concise hovers (no long multi-field label)
Legend:
- Grouped legend items for line + scatter so toggling hides both.
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource, HoverTool, Span, Range1d, LinearAxis,
    Legend, LegendItem
)

# ---------- Regex helpers ----------
P_LINE = re.compile(r'^(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*?VIDEOPOWEROPT:\s(?P<msg>.*)$')

P_IS_RUN = re.compile(r'runTimerTask\(', re.I)
P_MEAS_FPS  = re.compile(r'\bMeasured\s*fps\s*:\s*(?P<val>\d+(?:\.\d+)?)', re.I)
P_MEAS_UTIL = re.compile(r'\b(?:measuredUtil|measured\s*util)\s*:\s*(?P<val>\d+)', re.I)

P_THR    = re.compile(r'Current\s*dynamic\s*fps\s*control\s*threshold\s*:\s*(?P<th>\d+)', re.I)
P_HINT   = re.compile(r'\b(?:HintFps|Hint\s*Fps)\s*:\s*(?P<val>\d+(?:\.\d+)?)(?=[\s,.)]|$)', re.I)
P_TARGET = re.compile(r'\b(?:TargetFps|Target\s*Fps)\s*:\s*(?P<val>\d+(?:\.\d+)?)(?=[\s,.)]|$)', re.I)

P_SEND = re.compile(r'sendEvents', re.I)
P_HW_HINT    = re.compile(r'HWDecode\s*hint', re.I)
P_HW_DEFAULT = re.compile(r'HWDecode\s*default\s*hint', re.I)
P_CAM_PREVIEW = re.compile(r'Camera\s*preview\s*hint', re.I)  # NEW
P_CAM_CLOSE   = re.compile(r'Camera\s*close\s*hint', re.I)    # NEW

# collectAndDecide
P_COLLECT_AVG   = re.compile(r'collectAndDecide\(\).*?AvgFps\s*=\s*(?P<avgfps>\d+(?:\.\d+)?)\s*,\s*AvgUtil\s*=\s*(?P<avgutil>\d+(?:\.\d+)?)', re.I)
P_COLLECT_APPLY = re.compile(r'collectAndDecide\(\).*?Applied\s*power\s*profile\s*:\s*(?P<profile>-?\d+)\s*for\s*Camera\s*FPS\s*:\s*(?P<camfps>\d+(?:\.\d+)?)\s*,\s*UTIL\s*:\s*(?P<camutil>\d+(?:\.\d+)?)', re.I)

# processCameraEvent
P_PROC_CAM = re.compile(r'processCameraEvent\(\).*?Processing\s*event\s*:\s*(?P<event>-?\d+)\s*in\s*state\s*:\s*(?P<state>-?\d+)', re.I)

# other events
P_READ    = re.compile(r'readAndApplyProfile\(', re.I)
P_RELEASE = re.compile(r'releaseProfileLock\(', re.I)
P_TOUCH   = re.compile(r'Touch\s*Event\s*Received', re.I)

def parse_log(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore')
    rows, th_rows, events = [], [], []
    collect_rows, camstate_rows = [], []

    cur_year = datetime.now().year

    for line in text.splitlines():
        m = P_LINE.search(line)
        if not m:
            continue
        ts = datetime.strptime(f"{cur_year}-{m.group('ts')}", "%Y-%m-%d %H:%M:%S.%f")
        msg = m.group('msg').strip()

        # runTimerTask values (+ hints/targets if they appear here)
        if P_IS_RUN.search(msg):
            r = {'ts': ts, 'fps': None, 'util': None, 'hint': None, 'target': None}
            if (mf := P_MEAS_FPS.search(msg)):  r['fps'] = float(mf.group('val'))
            if (mu := P_MEAS_UTIL.search(msg)): r['util'] = float(mu.group('val'))
            if (mh := P_HINT.search(msg)):      r['hint'] = float(mh.group('val'))
            if (mt := P_TARGET.search(msg)):    r['target'] = float(mt.group('val'))
            rows.append(r)

        # threshold
        if (t := P_THR.search(msg)):
            th_rows.append({'ts': ts, 'threshold': float(t.group('th'))})

        # collectAndDecide variants
        if (ca := P_COLLECT_APPLY.search(msg)):
            collect_rows.append({'ts': ts,
                                 'profile': float(ca.group('profile')),
                                 'camfps': float(ca.group('camfps')),
                                 'camutil': float(ca.group('camutil'))})
        elif (av := P_COLLECT_AVG.search(msg)):
            collect_rows.append({'ts': ts,
                                 'profile': None,
                                 'camfps': float(av.group('avgfps')),
                                 'camutil': float(av.group('avgutil'))})

        # processCameraEvent
        if (pce := P_PROC_CAM.search(msg)):
            camstate_rows.append({'ts': ts,
                                  'camera_event': float(pce.group('event')),
                                  'camera_state': float(pce.group('state'))})

        # timeline events
        if P_SEND.search(msg):
            if P_HW_HINT.search(msg):    events.append({'ts': ts, 'event': 'HWDecode hint'})
            if P_HW_DEFAULT.search(msg): events.append({'ts': ts, 'event': 'HWDecode default hint'})
            if P_CAM_PREVIEW.search(msg):events.append({'ts': ts, 'event': 'Camera preview hint'})   # NEW
            if P_CAM_CLOSE.search(msg):  events.append({'ts': ts, 'event': 'Camera close hint'})     # NEW

        if P_READ.search(msg):    events.append({'ts': ts, 'event': 'readAndApplyProfile'})
        if P_RELEASE.search(msg): events.append({'ts': ts, 'event': 'releaseProfileLock'})
        if P_TOUCH.search(msg):   events.append({'ts': ts, 'event': 'Touch Event Received'})

    # DataFrames & aggregation
    fps_df = pd.DataFrame(rows)
    if not fps_df.empty:
        fps_df = fps_df.sort_values('ts').groupby('ts', as_index=False)\
                       .agg({'fps':'max','util':'max','hint':'max','target':'max'})

    thr_df = pd.DataFrame(th_rows).sort_values('ts') if th_rows else pd.DataFrame(columns=['ts','threshold'])
    evt_df = pd.DataFrame(events).sort_values('ts') if events else pd.DataFrame(columns=['ts','event'])

    col_df = pd.DataFrame(collect_rows)
    if not col_df.empty:
        col_df = col_df.sort_values('ts').groupby('ts', as_index=False)\
                       .agg({'profile':'max','camfps':'max','camutil':'max'})

    cam_df = pd.DataFrame(camstate_rows)
    if not cam_df.empty:
        cam_df = cam_df.sort_values('ts').groupby('ts', as_index=False)\
                       .agg({'camera_event':'max','camera_state':'max'})

    return fps_df, thr_df, evt_df, col_df, cam_df

def build_dataset(fps_df, thr_df, evt_df, col_df, cam_df):
    # unified time base
    for base_df in [fps_df, thr_df, col_df, cam_df]:
        if not base_df.empty:
            base = base_df[['ts']].drop_duplicates().copy()
            break
    else:
        base = pd.DataFrame({'ts': []})

    full = base.copy()
    for add_df, cols in [
        (thr_df, ['threshold']),
        (fps_df, ['fps','util','hint','target']),
        (col_df, ['profile','camfps','camutil']),
        (cam_df, ['camera_event','camera_state']),
    ]:
        if not add_df.empty:
            full = full.merge(add_df[['ts']+cols], on='ts', how='outer')

    full = full.sort_values('ts').reset_index(drop=True)

    # forward-fill state-like series so they appear as steps
    for col in ['hint','target','profile','camera_event','camera_state']:
        if col in full:
            full[col] = pd.to_numeric(full[col], errors='coerce')
            if full[col].notna().any():
                full[col] = full[col].ffill()

    # numeric coercion
    for col in ['fps','util','threshold','camfps','camutil']:
        if col in full:
            full[col] = pd.to_numeric(full[col], errors='coerce')

    return full, evt_df

def make_plot(full: pd.DataFrame, evt_df: pd.DataFrame, out_html: Path):
    output_file(str(out_html), title="Dynamic FPS + Camera – VIDEOPOWEROPT (Enhanced)")

    p = figure(x_axis_type='datetime', width=1600, height=950,
               title='Dynamic FPS + Camera behavior (VIDEOPOWEROPT) – Enhanced',
               tools='pan,wheel_zoom,box_zoom,reset,save')

    # Left axis
    p.yaxis.axis_label = 'FPS / Threshold / HintFps / TargetFps'

    # Right axis for UTIL + camera UTIL
    util_max = float(pd.to_numeric(full.get('util'), errors='coerce').max() or 0)
    camutil_max = float(pd.to_numeric(full.get('camutil'), errors='coerce').max() or 0)
    right_max = max(100, util_max, camutil_max) * 1.2
    p.extra_y_ranges = {"util": Range1d(start=0, end=right_max)}
    p.add_layout(LinearAxis(y_range_name='util', axis_label='Util / Camera UTIL'), 'right')

    # Extra range for camera event/state/profile
    cam_vals = []
    for col in ['camera_event','camera_state','profile']:
        if col in full and pd.to_numeric(full[col], errors='coerce').notna().any():
            cam_vals.append(float(pd.to_numeric(full[col], errors='coerce').max()))
    cam_max = max(cam_vals) if cam_vals else 0
    if cam_max:
        p.extra_y_ranges['cam'] = Range1d(start=-0.5, end=cam_max + 1.5)
        p.add_layout(LinearAxis(y_range_name='cam', axis_label='Camera event/state/profile'), 'left')

    src = ColumnDataSource(full)
    legend_items = []

    # Measured fps
    if 'fps' in full and full['fps'].notnull().any():
        r_fps_line = p.line('ts','fps', source=src, color="#1f77b4", line_width=2)
        r_fps_pts  = p.scatter('ts','fps', source=src, color="#1f77b4", size=6)
        legend_items.append(LegendItem(label='Measured fps', renderers=[r_fps_line, r_fps_pts]))
        p.add_tools(HoverTool(renderers=[r_fps_pts], tooltips=[('time','@ts{%F %T.%3N}'), ('fps','@fps{0.0}')],
                              formatters={'@ts':'datetime'}, mode='mouse'))

    # measuredUtil & Camera UTIL (right axis)
    if 'util' in full and full['util'].notnull().any():
        r_util_line = p.line('ts','util', source=src, color="#d62728", line_width=2, y_range_name='util')
        r_util_pts  = p.scatter('ts','util', source=src, color="#d62728", size=6, y_range_name='util')
        legend_items.append(LegendItem(label='measuredUtil', renderers=[r_util_line, r_util_pts]))
        p.add_tools(HoverTool(renderers=[r_util_pts], tooltips=[('time','@ts{%F %T.%3N}'), ('util','@util')],
                              formatters={'@ts':'datetime'}, mode='mouse'))

    #if 'camutil' in full and full['camutil'].notnull().any():
    #    r_cu_line = p.line('ts','camutil', source=src, color="#e377c2", line_width=2, y_range_name='util')
    #    r_cu_pts  = p.scatter('ts','camutil', source=src, color="#e377c2", size=6, y_range_name='util')
    #    legend_items.append(LegendItem(label='Camera UTIL (collectAndDecide)', renderers=[r_cu_line, r_cu_pts]))
    #    p.add_tools(HoverTool(renderers=[r_cu_pts], tooltips=[('time','@ts{%F %T.%3N}'), ('CamUtil','@camutil')],
    #                          formatters={'@ts':'datetime'}, mode='mouse'))

    # Threshold / HintFps / TargetFps
    if 'threshold' in full and full['threshold'].notnull().any():
        r_thr = p.step('ts','threshold', source=src, color="#2ca02c", line_width=2, mode='after', line_dash='dashed')
        legend_items.append(LegendItem(label='fps control threshold', renderers=[r_thr]))
        p.add_tools(HoverTool(renderers=[r_thr], tooltips=[('time','@ts{%F %T.%3N}'), ('threshold','@threshold{0}')],
                              formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))

    if 'hint' in full and full['hint'].notnull().any():
        r_hint = p.step('ts','hint', source=src, color="#ff7f0e", line_width=2, mode='after', line_dash='dotdash')
        legend_items.append(LegendItem(label='HintFps', renderers=[r_hint]))
        p.add_tools(HoverTool(renderers=[r_hint], tooltips=[('time','@ts{%F %T.%3N}'), ('HintFps','@hint{0.0}')],
                              formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))

    if 'target' in full and full['target'].notnull().any():
        r_target = p.step('ts','target', source=src, color="#9467bd", line_width=2, mode='after', line_dash='dotted')
        legend_items.append(LegendItem(label='TargetFps', renderers=[r_target]))
        p.add_tools(HoverTool(renderers=[r_target], tooltips=[('time','@ts{%F %T.%3N}'), ('TargetFps','@target{0.0}')],
                              formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))

    # Camera FPS (collectAndDecide)
    #if 'camfps' in full and full['camfps'].notnull().any():
    #    r_cf_line = p.line('ts','camfps', source=src, color="#8c564b", line_width=2)
    #    r_cf_pts  = p.scatter('ts','camfps', source=src, color="#8c564b", size=6)
    #    legend_items.append(LegendItem(label='Camera FPS (collectAndDecide)', renderers=[r_cf_line, r_cf_pts]))
    #    p.add_tools(HoverTool(renderers=[r_cf_pts], tooltips=[('time','@ts{%F %T.%3N}'), ('CamFPS','@camfps{0.00}')],
    #                          formatters={'@ts':'datetime'}, mode='mouse'))

    # Camera event/state/profile on dedicated 'cam' axis
    if 'cam' in p.extra_y_ranges:
        if 'camera_event' in full and full['camera_event'].notnull().any():
            r_evt = p.step('ts','camera_event', source=src, color="#17becf", line_width=2, mode='after', y_range_name='cam')
            legend_items.append(LegendItem(label='Camera Event', renderers=[r_evt]))
            p.add_tools(HoverTool(renderers=[r_evt], tooltips=[('time','@ts{%F %T.%3N}'), ('event','@camera_event{0}')],
                                  formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))
        if 'camera_state' in full and full['camera_state'].notnull().any():
            r_st = p.step('ts','camera_state', source=src, color="#bcbd22", line_width=2, mode='after', y_range_name='cam')
            legend_items.append(LegendItem(label='Camera State', renderers=[r_st]))
            p.add_tools(HoverTool(renderers=[r_st], tooltips=[('time','@ts{%F %T.%3N}'), ('state','@camera_state{0}')],
                                  formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))
        if 'profile' in full and full['profile'].notnull().any():
            r_prof = p.step('ts','profile', source=src, color="#7f7f7f", line_width=2, mode='after', y_range_name='cam')
            legend_items.append(LegendItem(label='Applied Profile', renderers=[r_prof]))
            p.add_tools(HoverTool(renderers=[r_prof], tooltips=[('time','@ts{%F %T.%3N}'), ('profile','@profile{0}')],
                                  formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))

    # Event markers at top of left axis
    left_max = 1.0
    for col in ['fps','threshold','hint','target','camfps']:
        if col in full and pd.to_numeric(full[col], errors='coerce').notna().any():
            left_max = max(left_max, float(pd.to_numeric(full[col], errors='coerce').max() or 0))
    evt_y = left_max * 1.15
    p.y_range.start = 0
    p.y_range.end = evt_y * 1.1

    if not evt_df.empty:
        evt_colors = {
            'readAndApplyProfile': '#ff7f0e',
            'releaseProfileLock':  '#9467bd',
            'Touch Event Received':'#17becf',
            'HWDecode hint':       '#e377c2',
            'HWDecode default hint':'#8c564b',
            'Camera preview hint': '#2ca02c',
            'Camera close hint':   '#d62728',
        }
        evt_markers = {
            'readAndApplyProfile': 'triangle',
            'releaseProfileLock':  'inverted_triangle',
            'Touch Event Received':'star',
            'HWDecode hint':       'diamond',
            'HWDecode default hint':'hex',
            'Camera preview hint': 'square',
            'Camera close hint':   'square_x',
        }
        for ev_type, grp in evt_df.groupby('event'):
            cds = ColumnDataSource({'ts': grp['ts'], 'y': [evt_y]*len(grp)})
            mrk = p.scatter('ts','y', source=cds, size=14,
                            color=evt_colors.get(ev_type,'gray'),
                            alpha=0.9, marker=evt_markers.get(ev_type,'circle_cross'))
            legend_items.append(LegendItem(label=ev_type, renderers=[mrk]))
            for t in grp['ts']:
                p.add_layout(Span(location=t.timestamp()*1000, dimension='height',
                                  line_color=evt_colors.get(ev_type,'gray'),
                                  line_dash='dotted', line_alpha=0.3))
            p.add_tools(HoverTool(renderers=[mrk],
                                  tooltips=[('event', ev_type), ('time','@ts{%F %T.%3N}')],
                                  formatters={'@ts':'datetime'}, mode='mouse'))

    # Explicit legend with grouped items so toggling hides all linked renderers
    legend = Legend(items=legend_items, location='top_left', click_policy='hide')
    p.add_layout(legend, 'right')
    save(p)

def main():
    ap = argparse.ArgumentParser(description="Plot Dynamic FPS + Camera from VIDEOPOWEROPT logs (Bokeh)")
    ap.add_argument('-i', '--input', required=True, type=Path, help='Path to logcat text file')
    ap.add_argument('-o', '--output', default=Path('dynamic_fps_plot.html'), type=Path,
                    help='Output HTML file (Bokeh)')
    args = ap.parse_args()

    fps_df, thr_df, evt_df, col_df, cam_df = parse_log(args.input)
    full_df, evt_df = build_dataset(fps_df, thr_df, evt_df, col_df, cam_df)
    make_plot(full_df, evt_df, args.output)
    print(f"Saved plot to: {args.output.resolve()}")

if __name__ == "__main__":
    main()