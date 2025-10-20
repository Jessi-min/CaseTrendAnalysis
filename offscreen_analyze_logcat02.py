# -*- coding: utf-8 -*-
"""
Dynamic FPS + Camera + Offscreen Power logcat plotter (Bokeh)

Usage:
  python plot_dynamic_fps_bokeh.py -i <logcat.txt> -o dynamic_fps_plot.html

Includes:
- VIDEOPOWEROPT:
  * Measured fps (left, line+scatter) and measuredUtil (right, line+scatter)
  * Threshold (dashed step), HintFps/TargetFps (step; robust spacing)
  * processCameraEvent -> Camera Event/State (step on camera axis)
  * collectAndDecide -> Applied Profile (step on camera axis)
  * sendEvents -> highlights for HWDecode hint/default hint, Camera preview/close hint
  * Camera FPS/UTIL (collectAndDecide) plotting is intentionally DISABLED (commented)

- OFFSCREENPOWEROPT:
  * workloadMonitor() -> movingAverage/1000 (line+scatter)
  * "workloadMonitor - release" markers with movingAverage/1000, latest WorkLoad, mWorkloadThreshold
  * runTimerTask() timer duration change markers (old→new)

- Concise per-series hovers; legend grouping (line+points) so toggling hides both
- Avoids W-1000 by adding a faint dummy renderer only if no data was drawn
- If both VIDEO and OFFSCREEN data found, renders Tabs (Video | Offscreen)
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource, HoverTool, Span, Range1d, LinearAxis,
    Legend, LegendItem, Tabs, Panel
)

# -----------------------------
# Common helpers
# -----------------------------
P_LINE_ANY = re.compile(
    r'^(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*?(?P<tag>VIDEOPOWEROPT|OFFSCREENPOWEROPT):\s(?P<msg>.*)$'
)

def _ts_with_year(ts_mmddhhmmssms: str) -> datetime:
    return datetime.strptime(f"{datetime.now().year}-{ts_mmddhhmmssms}", "%Y-%m-%d %H:%M:%S.%f")

def _ensure_has_renderer(p):
    # Add a faint dummy renderer if figure ended up empty
    has_renderers = any(getattr(r, "data_source", None) is not None for r in p.renderers)
    if not has_renderers:
        p.line([0, 1], [0, 0], color='#cccccc', alpha=0.1, legend_label='No data')
        p.title.text = (p.title.text or "Plot") + " — no valid data found"

# -----------------------------
# VIDEOPOWEROPT (based on your script)
# -----------------------------
# Regex patterns (whitespace tolerant; robust to spacing before/after colon)
P_LINE_V = re.compile(r'^(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*?VIDEOPOWEROPT:\s(?P<msg>.*)$')
P_IS_RUN = re.compile(r'runTimerTask\(', re.I)
P_MEAS_FPS  = re.compile(r'\bMeasured\s*fps\s*:\s*(?P<val>\d+(?:\.\d+)?)', re.I)
P_MEAS_UTIL = re.compile(r'\b(?:measuredUtil|measured\s*util)\s*:\s*(?P<val>\d+)', re.I)
P_THR       = re.compile(r'Current\s*dynamic\s*fps\s*control\s*threshold\s*:\s*(?P<th>\d+)', re.I)
P_HINT      = re.compile(r'\b(?:HintFps|Hint\s*Fps)\s*:\s*(?P<val>\d+(?:\.\d+)?)(?=[\s,.)]|$)', re.I)
P_TARGET    = re.compile(r'\b(?:TargetFps|Target\s*Fps)\s*:\s*(?P<val>\d+(?:\.\d+)?)(?=[\s,.)]|$)', re.I)
P_SEND      = re.compile(r'sendEvents', re.I)
P_HW_HINT   = re.compile(r'HWDecode\s*hint', re.I)
P_HW_DEF    = re.compile(r'HWDecode\s*default\s*hint', re.I)
P_CAM_PREV  = re.compile(r'Camera\s*preview\s*hint', re.I)     # NEW
P_CAM_CLOSE = re.compile(r'Camera\s*close\s*hint', re.I)       # NEW
P_COL_AVG   = re.compile(r'collectAndDecide\(\).*?AvgFps\s*=\s*(?P<avgfps>\d+(?:\.\d+)?)\s*,\s*AvgUtil\s*=\s*(?P<avgutil>\d+(?:\.\d+)?)', re.I)
P_COL_APPLY = re.compile(r'collectAndDecide\(\).*?Applied\s*power\s*profile\s*:\s*(?P<profile>-?\d+)\s*for\s*Camera\s*FPS\s*:\s*(?P<camfps>\d+(?:\.\d+)?)\s*,\s*UTIL\s*:\s*(?P<camutil>\d+(?:\.\d+)?)', re.I)
P_PROC_CAM  = re.compile(r'processCameraEvent\(\).*?Processing\s*event\s*:\s*(?P<event>-?\d+)\s*in\s*state\s*:\s*(?P<state>-?\d+)', re.I)

def parse_video(text: str):
    rows, th_rows, events = [], [], []
    collect_rows, camstate_rows = [], []

    for line in text.splitlines():
        m = P_LINE_V.search(line)
        if not m:
            continue
        ts = _ts_with_year(m.group('ts'))
        msg = m.group('msg').strip()

        # runTimerTask (fps/util + hint/target if present on same line)
        if P_IS_RUN.search(msg):
            r = {'ts': ts, 'fps': None, 'util': None, 'hint': None, 'target': None}
            if (mf := P_MEAS_FPS.search(msg)):  r['fps']    = float(mf.group('val'))
            if (mu := P_MEAS_UTIL.search(msg)): r['util']   = float(mu.group('val'))
            if (mh := P_HINT.search(msg)):      r['hint']   = float(mh.group('val'))
            if (mt := P_TARGET.search(msg)):    r['target'] = float(mt.group('val'))
            rows.append(r)

        # threshold
        if (t := P_THR.search(msg)):
            th_rows.append({'ts': ts, 'threshold': float(t.group('th'))})

        # collectAndDecide
        if (ca := P_COL_APPLY.search(msg)):
            collect_rows.append({'ts': ts,
                                 'profile': float(ca.group('profile')),
                                 'camfps': float(ca.group('camfps')),
                                 'camutil': float(ca.group('camutil'))})
        elif (av := P_COL_AVG.search(msg)):
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
            if P_HW_HINT.search(msg):  events.append({'ts': ts, 'event': 'HWDecode hint'})
            if P_HW_DEF.search(msg):   events.append({'ts': ts, 'event': 'HWDecode default hint'})
            if P_CAM_PREV.search(msg): events.append({'ts': ts, 'event': 'Camera preview hint'})
            if P_CAM_CLOSE.search(msg):events.append({'ts': ts, 'event': 'Camera close hint'})
        # you also keep these:
        if re.search(r'readAndApplyProfile\(', msg, re.I): events.append({'ts': ts, 'event': 'readAndApplyProfile'})
        if re.search(r'releaseProfileLock\(', msg, re.I):  events.append({'ts': ts, 'event': 'releaseProfileLock'})
        if re.search(r'Touch\s*Event\s*Received', msg, re.I): events.append({'ts': ts, 'event': 'Touch Event Received'})

    # DataFrames & aggregation
    fps_df = pd.DataFrame(rows)
    if not fps_df.empty:
        fps_df = fps_df.sort_values('ts').groupby('ts', as_index=False) \
                       .agg({'fps':'max', 'util':'max', 'hint':'max', 'target':'max'})

    thr_df = pd.DataFrame(th_rows).sort_values('ts') if th_rows else pd.DataFrame(columns=['ts','threshold'])
    evt_df = pd.DataFrame(events).sort_values('ts') if events else pd.DataFrame(columns=['ts','event'])

    col_df = pd.DataFrame(collect_rows)
    if not col_df.empty:
        col_df = col_df.sort_values('ts').groupby('ts', as_index=False) \
                       .agg({'profile':'max','camfps':'max','camutil':'max'})

    cam_df = pd.DataFrame(camstate_rows)
    if not cam_df.empty:
        cam_df = cam_df.sort_values('ts').groupby('ts', as_index=False) \
                       .agg({'camera_event':'max','camera_state':'max'})

    return fps_df, thr_df, evt_df, col_df, cam_df

def build_video_dataset(fps_df, thr_df, evt_df, col_df, cam_df):
    # unified time base (same order you had)
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

def plot_video(full: pd.DataFrame, evt_df: pd.DataFrame, title: str):
    p = figure(x_axis_type='datetime', width=1600, height=950,
               title=title, tools='pan,wheel_zoom,box_zoom,reset,save')
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

    # Measured fps (group line + points)
    if 'fps' in full and full['fps'].notnull().any():
        r_fps_line = p.line('ts','fps', source=src, color="#1f77b4", line_width=2)
        r_fps_pts  = p.scatter('ts','fps', source=src, color="#1f77b4", size=6)
        legend_items.append(LegendItem(label='Measured fps', renderers=[r_fps_line, r_fps_pts]))
        p.add_tools(HoverTool(renderers=[r_fps_pts],
                              tooltips=[('time','@ts{%F %T.%3N}'), ('fps','@fps{0.0}')],
                              formatters={'@ts':'datetime'}, mode='mouse'))

    # measuredUtil (RIGHT) (group line + points)
    if 'util' in full and full['util'].notnull().any():
        r_util_line = p.line('ts','util', source=src, color="#d62728", line_width=2, y_range_name='util')
        r_util_pts  = p.scatter('ts','util', source=src, color="#d62728", size=6, y_range_name='util')
        legend_items.append(LegendItem(label='measuredUtil', renderers=[r_util_line, r_util_pts]))
        p.add_tools(HoverTool(renderers=[r_util_pts],
                              tooltips=[('time','@ts{%F %T.%3N}'), ('util','@util')],
                              formatters={'@ts':'datetime'}, mode='mouse'))

    # Threshold / HintFps / TargetFps
    if 'threshold' in full and full['threshold'].notnull().any():
        r_thr = p.step('ts','threshold', source=src, color="#2ca02c", line_width=2, mode='after', line_dash='dashed')
        legend_items.append(LegendItem(label='fps control threshold', renderers=[r_thr]))
        p.add_tools(HoverTool(renderers=[r_thr],
                              tooltips=[('time','@ts{%F %T.%3N}'), ('threshold','@threshold{0}')],
                              formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))
    if 'hint' in full and full['hint'].notnull().any():
        r_hint = p.step('ts','hint', source=src, color="#ff7f0e", line_width=2, mode='after', line_dash='dotdash')
        legend_items.append(LegendItem(label='HintFps', renderers=[r_hint]))
        p.add_tools(HoverTool(renderers=[r_hint],
                              tooltips=[('time','@ts{%F %T.%3N}'), ('HintFps','@hint{0.0}')],
                              formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))
    if 'target' in full and full['target'].notnull().any():
        r_target = p.step('ts','target', source=src, color="#9467bd", line_width=2, mode='after', line_dash='dotted')
        legend_items.append(LegendItem(label='TargetFps', renderers=[r_target]))
        p.add_tools(HoverTool(renderers=[r_target],
                              tooltips=[('time','@ts{%F %T.%3N}'), ('TargetFps','@target{0.0}')],
                              formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))

    # Camera Event/State/Profile on 'cam' axis
    if 'cam' in p.extra_y_ranges:
        if 'camera_event' in full and full['camera_event'].notnull().any():
            r_evt = p.step('ts','camera_event', source=src, color="#17becf", line_width=2, mode='after', y_range_name='cam')
            legend_items.append(LegendItem(label='Camera Event', renderers=[r_evt]))
            p.add_tools(HoverTool(renderers=[r_evt],
                                  tooltips=[('time','@ts{%F %T.%3N}'), ('event','@camera_event{0}')],
                                  formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))
        if 'camera_state' in full and full['camera_state'].notnull().any():
            r_st = p.step('ts','camera_state', source=src, color="#bcbd22", line_width=2, mode='after', y_range_name='cam')
            legend_items.append(LegendItem(label='Camera State', renderers=[r_st]))
            p.add_tools(HoverTool(renderers=[r_st],
                                  tooltips=[('time','@ts{%F %T.%3N}'), ('state','@camera_state{0}')],
                                  formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))
        if 'profile' in full and full['profile'].notnull().any():
            r_prof = p.step('ts','profile', source=src, color="#7f7f7f", line_width=2, mode='after', y_range_name='cam')
            legend_items.append(LegendItem(label='Applied Profile', renderers=[r_prof]))
            p.add_tools(HoverTool(renderers=[r_prof],
                                  tooltips=[('time','@ts{%F %T.%3N}'), ('profile','@profile{0}')],
                                  formatters={'@ts':'datetime'}, mode='mouse', line_policy='nearest'))

    # Event markers at top of left axis
    left_max = 1.0
    for col in ['fps','threshold','hint','target','camfps']:  # camfps intentionally not drawn, but kept for range calc
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

    # Explicit legend so toggling hides all linked renderers
    legend = Legend(items=legend_items, location='top_left', click_policy='hide')
    p.add_layout(legend, 'right')
    _ensure_has_renderer(p)
    return p

# -----------------------------
# OFFSCREENPOWEROPT (NEW)
# -----------------------------
P_LINE_O = re.compile(r'^(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*?OFFSCREENPOWEROPT:\s(?P<msg>.*)$')
P_MA_SIMPLE = re.compile(r'workloadMonitor\(\).*?movingAverage\s*:\s*(?P<ma>\d+)', re.I)
P_RELEASE   = re.compile(r'workloadMonitor\s*-\s*release.*?movingAverage\s*:\s*(?P<ma>\d+)'
                         r'\s*,\s*latest\s*WorkLoad\s*:\s*(?P<wl>-?\d+)'
                         r'.*?mWorkloadThreshold\s*:\s*(?P<th>-?\d+)', re.I)
P_TIMER     = re.compile(r'runTimerTask\(\).*?Ready\s*to\s*update\s*timer\s*duration\s*:\s*(?P<dur>\d+)', re.I)

def parse_offscreen(text: str):
    rows_ma, release_events, timer_events = [], [], []
    last_timer = None

    for line in text.splitlines():
        m = P_LINE_O.search(line)
        if not m:
            continue
        ts = _ts_with_year(m.group('ts'))
        msg = m.group('msg').strip()

        if (s := P_MA_SIMPLE.search(msg)):
            rows_ma.append({'ts': ts, 'moving_avg': float(s.group('ma'))})

        if (r := P_RELEASE.search(msg)):
            ma = float(r.group('ma')); wl = float(r.group('wl')); th = float(r.group('th'))
            release_events.append({'ts': ts,
                                   'moving_avg_k': ma/1000.0,
                                   'latest_workload': wl,
                                   'mworkload_threshold': th})
            rows_ma.append({'ts': ts, 'moving_avg': ma})

        if (t := P_TIMER.search(msg)):
            dur = int(t.group('dur'))
            if last_timer is None or dur != last_timer:
                timer_events.append({'ts': ts, 'old': last_timer, 'new': dur})
                last_timer = dur

    ma_df = pd.DataFrame(rows_ma)
    if not ma_df.empty:
        ma_df = ma_df.sort_values('ts').groupby('ts', as_index=False) \
                     .agg({'moving_avg':'last'})
        ma_df['moving_avg_k'] = ma_df['moving_avg'] / 1000.0

    rel_df = pd.DataFrame(release_events).sort_values('ts') if release_events else pd.DataFrame(columns=['ts'])
    td_df  = pd.DataFrame(timer_events).sort_values('ts')  if timer_events  else pd.DataFrame(columns=['ts'])
    return ma_df, rel_df, td_df

def build_offscreen_dataset(ma_df, rel_df, td_df):
    base = ma_df[['ts']].drop_duplicates().copy() if not ma_df.empty else pd.DataFrame({'ts': []})
    full = base.copy()
    if not ma_df.empty:
        full = full.merge(ma_df[['ts','moving_avg_k']], on='ts', how='outer')
    full = full.sort_values('ts').reset_index(drop=True)
    return full, rel_df, td_df

def plot_offscreen(full: pd.DataFrame, rel_df: pd.DataFrame, td_df: pd.DataFrame, title: str):
    p = figure(x_axis_type='datetime', width=1600, height=900,
               title=title, tools='pan,wheel_zoom,box_zoom,reset,save')
    p.yaxis.axis_label = 'movingAverage/1000'
    src = ColumnDataSource(full)
    legend_items = []

    # movingAverage/1000
    if 'moving_avg_k' in full and full['moving_avg_k'].notnull().any():
        r_ma_line = p.line('ts','moving_avg_k', source=src, color="#1f77b4", line_width=2)
        r_ma_pts  = p.scatter('ts','moving_avg_k', source=src, color="#1f77b4", size=6)
        legend_items.append(LegendItem(label='movingAverage/1000', renderers=[r_ma_line, r_ma_pts]))
        p.add_tools(HoverTool(renderers=[r_ma_pts],
                              tooltips=[('time','@ts{%F %T.%3N}'), ('movingAvg/1000','@moving_avg_k{0.00}')],
                              formatters={'@ts':'datetime'}, mode='mouse'))

    # release markers
    if not rel_df.empty:
        top = (full['moving_avg_k'].max() * 1.15) if 'moving_avg_k' in full and full['moving_avg_k'].notnull().any() else 1.0
        rel_src = ColumnDataSource(rel_df.assign(y=top))
        r_rel = p.scatter('ts','y', source=rel_src, size=14, color="#ff7f0e", alpha=0.9, marker='square')
        legend_items.append(LegendItem(label='workloadMonitor - release', renderers=[r_rel]))
        p.add_tools(HoverTool(renderers=[r_rel],
                              tooltips=[('event','workloadMonitor - release'),
                                        ('movingAvg/1000','@moving_avg_k{0.00}'),
                                        ('latest WorkLoad','@latest_workload{0}'),
                                        ('mWorkloadThreshold','@mworkload_threshold{0}')],
                              formatters={'@ts':'datetime'}, mode='mouse'))
        for t in rel_df['ts']:
            p.add_layout(Span(location=t.timestamp()*1000, dimension='height',
                              line_color="#ff7f0e", line_dash='dotted', line_alpha=0.3))

    # timer duration change markers
    if not td_df.empty:
        top2 = (full['moving_avg_k'].max() * 1.25) if 'moving_avg_k' in full and full['moving_avg_k'].notnull().any() else 1.2
        td_src = ColumnDataSource(td_df.assign(y=top2))
        r_td = p.scatter('ts','y', source=td_src, size=14, color="#9467bd", alpha=0.9, marker='star')
        legend_items.append(LegendItem(label='Timer duration change', renderers=[r_td]))
        p.add_tools(HoverTool(renderers=[r_td],
                              tooltips=[('event','Timer duration change'), ('old (ms)','@old'), ('new (ms)','@new')],
                              formatters={'@ts':'datetime'}, mode='mouse'))
        for t in td_df['ts']:
            p.add_layout(Span(location=t.timestamp()*1000, dimension='height',
                              line_color="#9467bd", line_dash='dotted', line_alpha=0.3))

    # y-range for markers
    max_left = float(full['moving_avg_k'].max()) if 'moving_avg_k' in full and full['moving_avg_k'].notnull().any() else 1.0
    p.y_range.start = 0
    p.y_range.end = max_left * 1.35

    # Legend
    legend = Legend(items=legend_items, location='top_left', click_policy='hide')
    p.add_layout(legend, 'right')
    _ensure_has_renderer(p)
    return p

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot VIDEOPOWEROPT + OFFSCREENPOWEROPT logs (Bokeh)")
    ap.add_argument('-i', '--input', required=True, type=Path, help='Path to logcat text file')
    ap.add_argument('-o', '--output', default=Path('dynamic_fps_plot.html'), type=Path,
                    help='Output HTML file (Bokeh)')
    args = ap.parse_args()

    text = args.input.read_text(encoding='utf-8', errors='ignore')

    # Parse VIDEO and OFFSCREEN
    fps_df, thr_df, evt_df, col_df, cam_df = parse_video(text)     # from your base script  [1](https://qualcomm-my.sharepoint.com/personal/zehugong_qti_qualcomm_com/Documents/Microsoft%20Copilot%20Chat%20Files/Final_Video_DynamicFPS_3rdCamera_analyze.py)
    ma_df, rel_df, td_df = parse_offscreen(text)

    figs = []

    # VIDEO figure (only if data parsed)
    if not (fps_df.empty and thr_df.empty and evt_df.empty and col_df.empty and cam_df.empty):
        full_v, events_v = build_video_dataset(fps_df, thr_df, evt_df, col_df, cam_df)
        p_video = plot_video(full_v, events_v, title='Dynamic FPS + Camera (VIDEOPOWEROPT)')
        figs.append(Panel(child=p_video, title="Video"))

    # OFFSCREEN figure (only if data parsed)
    if not (ma_df.empty and rel_df.empty and td_df.empty):
        full_o, rel_o, td_o = build_offscreen_dataset(ma_df, rel_df, td_df)
        p_off = plot_offscreen(full_o, rel_o, td_o, title='Offscreen Workload (OFFSCREENPOWEROPT)')
        figs.append(Panel(child=p_off, title="Offscreen"))

    # Output
    output_file(str(args.output))
    if len(figs) == 0:
        # no data anywhere — produce an empty but valid figure
        p = figure(width=800, height=300, title="No VIDEO/OFFSCREEN data found")
        _ensure_has_renderer(p)
        save(p)
    elif len(figs) == 1:
        save(figs[0].child)
    else:
        save(Tabs(tabs=figs))

    print(f"Saved plot to: {args.output.resolve()}")

if __name__ == "__main__":
    main()