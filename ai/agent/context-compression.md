<style>
.ctxc-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:var(--cs-space-5);margin:var(--cs-space-7) 0}
.ctxc-grid-2{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.ctxc-card{--accent:var(--cs-color-brand);background:var(--cs-color-bg-soft);border:1px solid color-mix(in srgb,var(--accent) 18%,var(--cs-color-border));border-top:2px solid var(--accent);border-radius:var(--cs-radius-sm);padding:var(--cs-space-5) var(--cs-space-6)}
.ctxc-card h3{margin:0 0 var(--cs-space-3);font-size:var(--cs-text-md);line-height:var(--cs-leading-tight)}
.ctxc-card p,.ctxc-card ul{margin:0;color:var(--cs-color-text-muted);font-size:var(--cs-text-sm);line-height:var(--cs-leading-relaxed)}
.ctxc-card li+li{margin-top:var(--cs-space-2)}
.ctxc-card-s{--accent:var(--cs-color-success)}.ctxc-card-w{--accent:var(--cs-color-warning)}.ctxc-card-d{--accent:var(--cs-color-danger)}.ctxc-card-i{--accent:var(--cs-color-info)}
.ctxc-badge{display:inline-flex;border:1px solid var(--cs-color-border);border-radius:var(--cs-radius-xs);padding:0 var(--cs-space-2);font-family:var(--cs-font-mono);font-size:var(--cs-text-3xs);line-height:1.7;color:var(--cs-color-text-muted);white-space:nowrap}
.ctxc-badge-s{color:var(--cs-color-success);border-color:var(--cs-color-success)}.ctxc-badge-w{color:var(--cs-color-warning);border-color:var(--cs-color-warning)}.ctxc-badge-d{color:var(--cs-color-danger);border-color:var(--cs-color-danger)}.ctxc-badge-i{color:var(--cs-color-info);border-color:var(--cs-color-info)}
.ctxc-callout{--accent:var(--cs-color-brand);border-left:3px solid var(--accent);background:var(--cs-color-bg-soft);border-radius:0 var(--cs-radius-sm) var(--cs-radius-sm) 0;padding:var(--cs-space-5) var(--cs-space-6);margin:var(--cs-space-7) 0;color:var(--cs-color-text-muted);font-size:var(--cs-text-sm)}
.ctxc-callout-w{--accent:var(--cs-color-warning)}.ctxc-callout-d{--accent:var(--cs-color-danger)}.ctxc-callout strong{color:var(--cs-color-text)}
.ctxc-table-wrap{overflow-x:auto;border:1px solid color-mix(in srgb,var(--cs-color-text) 12%,var(--cs-color-border));border-radius:var(--cs-radius-xs);background:color-mix(in srgb,var(--cs-color-brand) 2%,var(--cs-color-bg-soft));margin:var(--cs-space-7) 0}
.ctxc-table{min-width:680px;width:100%;margin:0;border-collapse:collapse;font-size:var(--cs-text-xs)}
.ctxc-table th,.ctxc-table td{padding:var(--cs-space-3) var(--cs-space-4);border-right:1px solid color-mix(in srgb,var(--cs-color-text) 7%,transparent);border-bottom:1px solid color-mix(in srgb,var(--cs-color-text) 7%,transparent);text-align:left;vertical-align:top;line-height:1.55}
.ctxc-table th:last-child,.ctxc-table td:last-child{border-right:0}
.ctxc-table th{background:var(--cs-color-bg-elevated);color:var(--cs-color-text);font-family:var(--cs-font-mono);font-size:var(--cs-text-3xs);font-weight:600;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap;border-bottom-color:color-mix(in srgb,var(--cs-color-brand) 28%,var(--cs-color-border))}
.ctxc-table td{color:var(--cs-color-text-muted)}
.ctxc-table td:first-child{color:var(--cs-color-text);font-weight:500}
.ctxc-table tbody tr:nth-child(even){background:color-mix(in srgb,var(--cs-color-brand) 2.5%,transparent)}
.ctxc-table tbody tr:hover{background:color-mix(in srgb,var(--cs-color-brand) 6%,transparent)}
.ctxc-table tr:last-child td{border-bottom:none}.ctxc-mono{font-family:var(--cs-font-mono);font-size:var(--cs-text-xs);white-space:nowrap}
.ctxc-table code{padding:.05rem .28rem;border-radius:var(--cs-radius-xs);background:color-mix(in srgb,var(--cs-color-brand) 8%,var(--cs-color-bg));color:var(--cs-color-brand);font-family:var(--cs-font-mono);font-size:calc(1em - 1px);white-space:nowrap}
.ctxc-figure{margin:var(--cs-space-8) 0}.ctxc-figure-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:var(--cs-space-3);align-items:baseline;margin-bottom:var(--cs-space-4)}
.ctxc-figure-title{margin:0;font-size:var(--cs-text-lg);font-weight:var(--cs-font-weight-bold)}.ctxc-figure-meta{margin:0;font-family:var(--cs-font-mono);font-size:var(--cs-text-3xs);letter-spacing:.08em;color:var(--cs-color-text-subtle)}
.ctxc-scroll{overflow-x:auto;overflow-y:hidden;scrollbar-width:thin}.ctxc-note{margin:var(--cs-space-4) 0 0;color:var(--cs-color-text-muted);font-size:var(--cs-text-xs);line-height:var(--cs-leading-relaxed)}
.ctxc-svg{display:block;width:100%;height:auto;min-width:0}.ctxc-svg text{font-family:inherit}.ctxc-svg .mono{font-family:var(--cs-font-mono)}
.ctxc-gridline{stroke:var(--cs-color-border);stroke-width:1;stroke-dasharray:4 6;opacity:.72}.ctxc-axis{stroke:var(--cs-color-border);stroke-width:1}.ctxc-label{fill:var(--cs-color-text-subtle);font-size:11px}.ctxc-text{fill:var(--cs-color-text-muted);font-size:12px}.ctxc-title{fill:var(--cs-color-text);font-size:14px;font-weight:600}
.ctxc-line{stroke:var(--cs-color-brand);stroke-width:2.5;fill:none}.ctxc-line-w{stroke:var(--cs-color-warning);stroke-width:2;fill:none}.ctxc-line-d{stroke:var(--cs-color-danger);stroke-width:2;stroke-dasharray:6 5;fill:none}
.ctxc-area{fill:color-mix(in srgb,var(--cs-color-brand) 15%,transparent)}.ctxc-point{fill:var(--cs-color-bg-soft);stroke:var(--cs-color-brand);stroke-width:2}.ctxc-m-b{fill:var(--cs-color-brand)}.ctxc-m-w{fill:var(--cs-color-warning)}.ctxc-m-d{fill:var(--cs-color-danger)}.ctxc-m-t{fill:var(--cs-color-on-brand);font-family:var(--cs-font-mono);font-size:10px;font-weight:600}
.ctxc-bar-s{fill:var(--cs-color-success)}.ctxc-bar-d{fill:var(--cs-color-danger)}.ctxc-box{fill:var(--cs-color-bg-elevated);stroke:var(--cs-color-border)}.ctxc-box-s{fill:color-mix(in srgb,var(--cs-color-success) 13%,transparent);stroke:var(--cs-color-success)}.ctxc-box-w{fill:color-mix(in srgb,var(--cs-color-warning) 13%,transparent);stroke:var(--cs-color-warning)}.ctxc-box-d{fill:color-mix(in srgb,var(--cs-color-danger) 13%,transparent);stroke:var(--cs-color-danger)}.ctxc-box-i{fill:color-mix(in srgb,var(--cs-color-info) 13%,transparent);stroke:var(--cs-color-info)}
.ctxc-arrow{stroke:var(--cs-color-brand);stroke-width:1.25;fill:none;marker-end:url(#ctxc-arrow);opacity:.72}.ctxc-arrow-head{fill:var(--cs-color-brand)}.ctxc-threshold{fill:var(--cs-color-text-muted);font-size:11px;paint-order:stroke;stroke:var(--cs-color-bg);stroke-width:5px;stroke-linejoin:round}
.ctxc-code{margin:var(--cs-space-6) 0;border:1px solid var(--cs-color-border-strong);border-left:3px solid var(--cs-color-warning);border-radius:0 var(--cs-radius-sm) var(--cs-radius-sm) 0;background:var(--cs-color-bg-elevated);padding:var(--cs-space-5) var(--cs-space-6);overflow-x:auto;font-family:var(--cs-font-mono);font-size:var(--cs-text-xs);line-height:var(--cs-leading-relaxed);color:var(--cs-color-text-muted);white-space:pre}
.vp-doc .ctxc-table{margin:0}.vp-doc .ctxc-note,.vp-doc .ctxc-figure-title,.vp-doc .ctxc-figure-meta{margin-top:0}.vp-doc .ctxc-figure-title{margin-bottom:0}
.ctxc details{border:1px solid color-mix(in srgb,var(--cs-color-brand) 18%,var(--cs-color-border));border-left:3px solid var(--cs-color-brand);border-radius:var(--cs-radius-sm);background:var(--cs-color-bg-soft);margin:var(--cs-space-3) 0}.ctxc details summary{cursor:pointer;font-weight:600;padding:var(--cs-space-4) var(--cs-space-5);background:color-mix(in srgb,var(--cs-color-brand) 5%,var(--cs-color-bg-soft))}.ctxc details p{margin:0;padding:0 var(--cs-space-5) var(--cs-space-5) var(--cs-space-9);color:var(--cs-color-text-muted);font-size:var(--cs-text-sm);line-height:var(--cs-leading-relaxed)}
@media(max-width:640px){.ctxc-figure-head{display:block}.ctxc-figure-meta{display:block;margin-top:var(--cs-space-2)}.ctxc-svg{min-width:680px}}
.vp-doc table.ctxc-table{min-width:680px;width:100%;margin:0;border-collapse:collapse;font-size:var(--cs-text-xs)}
.vp-doc table.ctxc-table th,.vp-doc table.ctxc-table td{padding:var(--cs-space-3) var(--cs-space-4);border-right:1px solid color-mix(in srgb,var(--cs-color-text) 7%,transparent);border-bottom:1px solid color-mix(in srgb,var(--cs-color-text) 7%,transparent);line-height:1.55}
.vp-doc table.ctxc-table th{padding:var(--cs-space-3) var(--cs-space-4);background:var(--cs-color-bg-elevated);font-family:var(--cs-font-mono);font-size:var(--cs-text-3xs);font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.vp-doc table.ctxc-table td{font-size:var(--cs-text-xs)}
.vp-doc table.ctxc-table tbody tr:nth-child(even){background:color-mix(in srgb,var(--cs-color-brand) 2.5%,transparent)}
.vp-doc table.ctxc-table tbody tr:hover{background:color-mix(in srgb,var(--cs-color-brand) 6%,transparent)}
.vp-doc table.ctxc-table code{padding:.05rem .28rem;border-radius:var(--cs-radius-xs);background:color-mix(in srgb,var(--cs-color-brand) 8%,var(--cs-color-bg));color:var(--cs-color-brand);font-family:var(--cs-font-mono);font-size:calc(1em - 1px);white-space:nowrap}
.ctxc-flow{margin:var(--cs-space-7) 0;display:grid;grid-template-columns:minmax(0,1fr) 36px minmax(0,1fr) 36px minmax(0,1fr) 36px minmax(0,1fr);align-items:stretch}
.ctxc-mobile-loop{display:none}
.ctxc-flow-loop-back{display:none}
.ctxc-flow{position:relative}
.ctxc-flow::before{content:"";position:absolute;left:16px;right:16px;bottom:-24px;height:22px;border-left:2px dashed var(--cs-color-text-subtle);border-right:2px dashed var(--cs-color-text-subtle);border-bottom:2px dashed var(--cs-color-text-subtle);border-radius:0 0 var(--cs-radius-sm) var(--cs-radius-sm);pointer-events:none}
.ctxc-flow::after{content:"新事件追加回原始档案";position:absolute;left:50%;bottom:-32px;transform:translateX(-50%);font-family:var(--cs-font-mono);font-size:var(--cs-text-3xs);color:var(--cs-color-text-subtle);white-space:nowrap;background:var(--cs-color-bg);padding:0 var(--cs-space-2)}
.ctxc-flow-node{--flow-color:var(--cs-color-brand);min-height:190px;padding:var(--cs-space-5);border:0;border-top:2px solid var(--flow-color);border-radius:0 0 var(--cs-radius-sm) var(--cs-radius-sm);background:color-mix(in srgb,var(--flow-color) 7%,var(--cs-color-bg-soft))}
.ctxc-flow-node h3{margin:0 0 var(--cs-space-2);font-size:var(--cs-text-sm);line-height:var(--cs-leading-tight)}
.ctxc-flow-kicker{margin:0 0 var(--cs-space-3);font-family:var(--cs-font-mono);font-size:9px;letter-spacing:.04em;color:var(--cs-color-text-subtle);white-space:nowrap}
.ctxc-flow-node p{margin:0 0 var(--cs-space-3);color:var(--cs-color-text-muted);font-size:var(--cs-text-xs);line-height:1.65}
.ctxc-flow-node ul{margin:0;padding-left:var(--cs-space-5);color:var(--cs-color-text-muted);font-size:var(--cs-text-3xs);line-height:1.65}
.ctxc-flow-node li+li{margin-top:2px}
.ctxc-flow-log{--flow-color:var(--cs-color-info)}.ctxc-flow-work{--flow-color:var(--cs-color-warning)}.ctxc-flow-payload{--flow-color:var(--cs-color-success)}
.ctxc-flow-arrow{position:relative;min-height:190px;display:flex;align-items:center;justify-content:center}.ctxc-flow-arrow svg{display:block;width:24px;height:24px;overflow:visible;stroke:var(--cs-color-text-subtle);stroke-width:2;fill:none}
.ctxc-flow-bypass{margin-top:var(--cs-space-3);padding:var(--cs-space-4) var(--cs-space-5);border-left:2px solid var(--cs-color-info);background:color-mix(in srgb,var(--cs-color-info) 6%,var(--cs-color-bg-soft));color:var(--cs-color-text-muted);font-size:var(--cs-text-xs);line-height:1.65}
.ctxc-flow-bypass strong{color:var(--cs-color-text)}
.ctxc-flow-loop{margin:var(--cs-space-3) 0 0;padding:var(--cs-space-4) var(--cs-space-5);border:1px dashed color-mix(in srgb,var(--cs-color-brand) 36%,var(--cs-color-border));border-radius:var(--cs-radius-sm);color:var(--cs-color-text-muted);font-size:var(--cs-text-xs);line-height:1.65}
.ctxc-flow-loop{margin-top:48px}
.ctxc-flow-loop strong{color:var(--cs-color-text)}
.vp-doc .ctxc-flow-node h3,.vp-doc .ctxc-flow-node .ctxc-flow-kicker,.vp-doc .ctxc-flow-node p,.vp-doc .ctxc-flow-node ul{margin-top:0}.vp-doc .ctxc-flow-node h3{margin-bottom:var(--cs-space-1)}.vp-doc .ctxc-flow-node p{margin-bottom:var(--cs-space-3)}
@media(max-width:820px){
	/* 移动端四角闭环 1→2→3→4→1；槽位占满卡片列，flex 居中，箭头自动对齐列中心 */
	.ctxc-flow{position:relative;grid-template-columns:minmax(0,1fr) 44px minmax(0,1fr);grid-template-rows:auto auto auto;gap:var(--cs-space-2) 0;padding-left:0}
	.ctxc-flow::before,.ctxc-flow::after{display:none}
	.ctxc-mobile-loop{display:none}
	.ctxc-flow-node{position:relative;display:flex;flex-direction:column;min-height:0;margin:0;padding:var(--cs-space-4);border-top:0;border-left:3px solid var(--flow-color);border-radius:0 var(--cs-radius-sm) var(--cs-radius-sm) 0}
	.ctxc-flow-node::before,.ctxc-flow-node::after{display:none}
	.ctxc-flow-node.ctxc-flow-log{grid-column:1;grid-row:1}
	.ctxc-flow-node.ctxc-flow-work{grid-column:3;grid-row:1}
	.ctxc-flow-node.ctxc-flow-payload{grid-column:3;grid-row:3}
	.ctxc-flow-node.ctxc-flow-action{grid-column:1;grid-row:3}
	.ctxc-flow-kicker{margin-bottom:var(--cs-space-2);font-size:9px;white-space:normal;line-height:1.35}
	.ctxc-flow-node p{margin-bottom:var(--cs-space-2);font-size:var(--cs-text-3xs);line-height:1.45}
	.ctxc-flow-node ul{display:flex;flex-direction:column;align-items:flex-start;flex-wrap:nowrap;gap:0;margin-top:auto;padding-top:var(--cs-space-3);padding-left:0;list-style:none}
	.vp-doc .ctxc-flow-node li,.vp-doc .ctxc-flow-node ul li{width:100%;margin:0;padding:4px 8px;border-radius:var(--cs-radius-xs);background:color-mix(in srgb,var(--flow-color) 9%,var(--cs-color-bg));font-size:9px;line-height:1.35}
	.vp-doc .ctxc-flow-node li+li{margin-top:4px}
	/* 箭头槽：横箭头占中列；竖箭头占满对应卡片列，flex 居中到列中心 */
	.ctxc-flow-arrow{display:flex;align-items:center;justify-content:center;width:100%;min-height:36px;margin:0}
	.ctxc-flow-arrow svg{display:block;width:22px;height:22px;overflow:visible;stroke:var(--cs-color-brand);stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round;fill:none}
	.ctxc-flow-arrow:nth-child(4){grid-column:2;grid-row:1;min-height:100%;width:44px}.ctxc-flow-arrow:nth-child(4) svg{transform:none}
	.ctxc-flow-arrow:nth-child(6){grid-column:3;grid-row:2}.ctxc-flow-arrow:nth-child(6) svg{transform:rotate(90deg)}
	.ctxc-flow-arrow:nth-child(8){grid-column:2;grid-row:3;width:44px}.ctxc-flow-arrow:nth-child(8) svg{transform:rotate(180deg)}
	/* 左 ↑ 回写(4→1)：占满左列行间隙，flex 自动对齐左卡中心 */
	.ctxc-flow-loop-back{display:flex;align-items:center;justify-content:center;grid-column:1;grid-row:2;width:100%;min-height:36px;margin:0}
	.ctxc-flow-loop-back svg{display:block;width:22px;height:22px;overflow:visible;stroke:var(--cs-color-brand);stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round;fill:none}
	.ctxc-flow-loop{position:relative;margin-top:var(--cs-space-4);padding:var(--cs-space-4) var(--cs-space-5);border:0;border-left:3px solid var(--cs-color-brand);border-radius:0 var(--cs-radius-sm) var(--cs-radius-sm) 0;background:color-mix(in srgb,var(--cs-color-brand) 8%,var(--cs-color-bg-soft));color:var(--cs-color-text-muted);font-size:var(--cs-text-3xs);line-height:1.55}
	.ctxc-flow-bypass{padding:var(--cs-space-4);font-size:var(--cs-text-3xs);line-height:1.55}
	.ctxc-flow-loop::before{content:"↻ 事件回写";display:block;margin-bottom:var(--cs-space-2);color:var(--cs-color-brand);font-family:var(--cs-font-mono);font-size:var(--cs-text-sm);font-weight:600;line-height:1}
}
.ctxc-result{margin:var(--cs-space-7) 0}
.ctxc-result-before{margin-bottom:0}
.ctxc-result-before-head{display:flex;align-items:baseline;justify-content:space-between;gap:var(--cs-space-3);margin-bottom:var(--cs-space-2);padding:0 2px}
.ctxc-result-before-title{font-family:var(--cs-font-mono);font-size:var(--cs-text-3xs);letter-spacing:.08em;color:var(--cs-color-text-subtle)}
.ctxc-result-before-size{font-family:var(--cs-font-mono);font-size:var(--cs-text-3xs);color:var(--cs-color-text-subtle)}
.ctxc-result-before .ctxc-win{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--cs-space-3);margin:0;flex-direction:row}
.ctxc-result-arrow{display:flex;align-items:center;justify-content:center;height:30px;color:var(--cs-color-brand);font-family:var(--cs-font-mono);font-size:18px;line-height:1}
.ctxc-result-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--cs-space-3)}
.ctxc-result-card{--rc:var(--cs-color-brand);border-top:2px solid var(--rc);border-radius:0 0 var(--cs-radius-sm) var(--cs-radius-sm);background:color-mix(in srgb,var(--rc) 6%,var(--cs-color-bg-soft));padding:var(--cs-space-4);display:flex;flex-direction:column}
.ctxc-result-card-name{font-family:var(--cs-font-mono);font-size:var(--cs-text-xs);font-weight:700;color:var(--rc);margin-bottom:2px}
.ctxc-result-card-size{font-family:var(--cs-font-mono);font-size:var(--cs-text-3xs);color:var(--cs-color-text-subtle);margin-bottom:var(--cs-space-3)}
.ctxc-result-card p{margin:0 0 var(--cs-space-2);color:var(--cs-color-text-muted);font-size:var(--cs-text-3xs);line-height:1.5}
.ctxc-result-card ul{margin:0 auto 0 0;padding-left:0;list-style:none;display:flex;flex-direction:column;gap:3px;margin-top:auto;padding-top:var(--cs-space-2);width:100%}
.ctxc-result-card li{padding:3px 7px;border-radius:var(--cs-radius-xs);background:color-mix(in srgb,var(--rc) 9%,var(--cs-color-bg));font-size:9px;line-height:1.35;color:var(--cs-color-text-muted)}
.ctxc-result-card li.keep{color:var(--cs-color-text)}
.ctxc-win{display:flex;flex-direction:column;gap:4px;margin-top:var(--cs-space-3)}
.ctxc-win-block{border:1px solid var(--rc);border-radius:var(--cs-radius-xs);padding:var(--cs-space-2) var(--cs-space-3);font-size:9px;line-height:1.35;color:var(--cs-color-text-muted);position:relative}
.ctxc-win-block b{display:block;font-weight:600;color:var(--cs-color-text);font-size:10px;margin-bottom:1px}
.ctxc-win-block .tok{position:absolute;top:var(--cs-space-2);right:var(--cs-space-3);font-family:var(--cs-font-mono);font-size:9px;color:var(--cs-color-text-subtle)}
.ctxc-win-block.is-keep{background:color-mix(in srgb,var(--rc) 10%,var(--cs-color-bg));border-color:var(--rc)}
.ctxc-win-block.is-replace{background:color-mix(in srgb,var(--rc) 4%,var(--cs-color-bg));border-style:dashed;border-color:color-mix(in srgb,var(--rc) 45%,var(--cs-color-border))}
.ctxc-win-block.is-new{background:color-mix(in srgb,var(--rc) 8%,var(--cs-color-bg));border-color:color-mix(in srgb,var(--rc) 70%,var(--cs-color-border))}
.ctxc-win-block.is-gone{opacity:.4;background:var(--cs-color-bg-soft);border-style:solid;border-color:var(--cs-color-border)}
.ctxc-result-before .ctxc-win-block{min-height:62px}
.ctxc-win-label{font-family:var(--cs-font-mono);font-size:8px;letter-spacing:.06em;color:var(--cs-color-text-subtle);margin:var(--cs-space-1) 0 0}
.ctxc-win-arrow{display:flex;align-items:center;justify-content:center;color:var(--rc);font-family:var(--cs-font-mono);font-size:14px;line-height:1;margin:var(--cs-space-1) 0}
.rc-blob{--rc:var(--cs-color-brand)}.rc-visible{--rc:var(--cs-color-warning)}.rc-soft{--rc:var(--cs-color-success)}.rc-event{--rc:var(--cs-color-info)}
@media(max-width:820px){.ctxc-result-grid{grid-template-columns:1fr 1fr}.ctxc-result-before-list{grid-template-columns:1fr 1fr}}
@media(max-width:480px){.ctxc-result-grid{grid-template-columns:1fr}.ctxc-result-before-list{grid-template-columns:1fr}}
</style>

# Agent 上下文压缩

Context 是一次模型调用实际发送的 token 序列，在每次请求时重新构造；memory 保存绑定主体、可随交互更新的状态，knowledge 保存按权限共享、经版本更新的资料，二者都在窗口外持久化，经召回和装配进入本轮输入。

长任务 Agent 会把用户消息、工具调用、观察结果和外部状态持续追加进历史。上下文压缩处理的不是磁盘上的完整历史，而是**下一轮请求中哪些内容以何种形式出现**：保持原文、降级为恢复句柄、读时投影，或固化到窗口外。成熟系统同时保存完整事件、维护带标记的工作集、临时组装请求载荷，并让工具返回值、外置记忆、读时投影、摘要和 KV Cache 策略协同工作。

---

## 1. 压缩的对象：从完整历史到请求载荷

该过程构成一个持续追加的闭环：完整事件先进入持久记录，再经元数据筛选形成工作集，并在每次请求前投影为模型输入；模型输出与工具执行产生新事件，随后追加回持久记录并进入下一轮。

<div class="ctxc-flow">
	<div class="ctxc-mobile-loop" aria-hidden="true"><svg viewBox="0 0 342 486" preserveAspectRatio="none">
		<defs>
			<marker id="m-top" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="#2f6fd0"></path></marker>
			<marker id="m-right" markerWidth="10" markerHeight="10" refX="5" refY="2" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="#2f6fd0"></path></marker>
			<marker id="m-bottom" markerWidth="10" markerHeight="10" refX="2" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="#2f6fd0"></path></marker>
			<marker id="m-left" markerWidth="10" markerHeight="10" refX="5" refY="8" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="#2f6fd0"></path></marker>
		</defs>
		<!-- 上 1→2（卡片右缘 151 → 左缘 191，中线 y=111） -->
		<path d="M154 111 H179" fill="none" stroke="#2f6fd0" stroke-width="2.4" marker-end="url(#m-top)"></path>
		<!-- 右 2→3（中槽 x≈171，行间隙 222→262） -->
		<path d="M171 226 V255" fill="none" stroke="#2f6fd0" stroke-width="2.4" marker-end="url(#m-right)"></path>
		<!-- 下 3→4（卡片左缘 191 → 右缘 151，中线 y≈374） -->
		<path d="M188 374 H163" fill="none" stroke="#2f6fd0" stroke-width="2.4" marker-end="url(#m-bottom)"></path>
		<!-- 左 4→1 事件回写：在左列右缘与中槽之间竖直向上，虚线表示回写 -->
		<path d="M139 258 V205" fill="none" stroke="#2f6fd0" stroke-width="2.4" stroke-dasharray="5 4" marker-end="url(#m-left)"></path>
	</svg></div>
	<div class="ctxc-flow-loop-back" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M12 21 V5 M7 10 L12 4 L17 10"></path></svg></div>
	<section class="ctxc-flow-node ctxc-flow-log">
		<h3>1 · 持久事件记录</h3>
		<div class="ctxc-flow-kicker">transcript / event log</div>
		<p>按时间顺序保存完整交互事实，是审计、回放与恢复的依据。</p>
		<ul>
			<li>消息、工具调用与完整观察</li>
			<li>append-only，不就地改写</li>
			<li>独立于本轮模型可见性</li>
		</ul>
	</section>
	<div class="ctxc-flow-arrow"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12 H19 M15 7 L21 12 L15 17"></path></svg></div>
	<section class="ctxc-flow-node ctxc-flow-work">
		<h3>2 · 工作集元数据</h3>
		<div class="ctxc-flow-kicker">annotated working set</div>
		<p>在不改变原始事实的前提下，为消息附加生命周期与可见性元数据。</p>
		<ul>
			<li><code>pinned</code>：强制保留</li>
			<li><code>cleared</code>：正文移出工作集</li>
			<li><code>recovery</code>：记录恢复句柄</li>
		</ul>
	</section>
	<div class="ctxc-flow-arrow"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12 H19 M15 7 L21 12 L15 17"></path></svg></div>
	<section class="ctxc-flow-node ctxc-flow-payload">
		<h3>3 · 请求载荷投影</h3>
		<div class="ctxc-flow-kicker">per-request payload</div>
		<p>根据工作集投影生成的单轮模型输入，定义本轮可见信息及其顺序。</p>
		<ul>
			<li>系统规则、消息、附件与摘要</li>
			<li>大结果仅保留结论和入口</li>
			<li>请求前构建，请求后失效</li>
		</ul>
	</section>
	<div class="ctxc-flow-arrow"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12 H19 M15 7 L21 12 L15 17"></path></svg></div>
	<section class="ctxc-flow-node ctxc-flow-action">
		<h3>4 · 决策执行与事件生成</h3>
		<div class="ctxc-flow-kicker">action and observation</div>
		<p>模型基于本轮载荷生成决策；决策执行后产生新的消息、状态和观察。</p>
		<ul>
			<li>文本响应或工具调用</li>
			<li>外部环境状态变化</li>
			<li>新事件追加并进入下一轮</li>
		</ul>
	</section>
</div>

<div class="ctxc-flow-loop"><strong>闭环 · 事件回写：</strong>模型输出、工具调用及其观察结果不是终端产物，而是下一轮的新增事件。它们追加回第 1 层「持久事件记录」，重新经过元数据标记、载荷投影与请求组装；压缩主要作用于工作集到请求载荷的转换。</div>

<div class="ctxc-flow-bypass"><strong>旁路存储：</strong>代码、报告、日志和结构化状态等持久产物独立保存，不随每轮消息线性传递。只有在任务需要时，它们才通过路径、URL 或附件进入本轮请求载荷。</div>

因此，清理通常不是删除事实，而是改变事实在下一轮请求中的表示形式：保留原文、替换为占位符、关联外置产物，或改写为局部摘要。读时投影类似数据库视图：底层事件不删除，模型请求发生时才计算可见载荷。这使事实保存、上下文选择和历史回滚相互解耦。

<div class="ctxc-callout"><strong>核心定义：</strong>压缩是在保持任务可继续的前提下，降低下一次请求工作集的噪声、体积和缓存成本；删除只是其中一种实现，而且通常不是首选。</div>

<div class="ctxc-figure">
	<div class="ctxc-figure-head"><h3 class="ctxc-figure-title">长会话中的 token 压力曲线</h3><p class="ctxc-figure-meta">SAWTOOTH / 示意，不代表实测</p></div>
	<div class="ctxc-scroll"><svg class="ctxc-svg" viewBox="0 0 1000 430" role="img" aria-labelledby="saw-title saw-desc">
		<title id="saw-title">活动上下文随清理和压缩形成锯齿形变化</title><desc id="saw-desc">token 在阶段内上升，遇到清理或摘要阈值后下降，最近工作保持较高保真度。</desc>
		<line x1="760" y1="28" x2="790" y2="28" class="ctxc-line"></line><text x="798" y="32" class="ctxc-label">活跃上下文 token</text>
		<line x1="70" y1="70" x2="950" y2="70" class="ctxc-gridline"></line><line x1="70" y1="135" x2="950" y2="135" class="ctxc-gridline"></line><line x1="70" y1="200" x2="950" y2="200" class="ctxc-gridline"></line><line x1="70" y1="265" x2="950" y2="265" class="ctxc-gridline"></line><line x1="70" y1="330" x2="950" y2="330" class="ctxc-axis"></line>
		<text x="46" y="74" class="ctxc-label mono" text-anchor="end">200</text><text x="46" y="139" class="ctxc-label mono" text-anchor="end">150</text><text x="46" y="204" class="ctxc-label mono" text-anchor="end">100</text><text x="46" y="269" class="ctxc-label mono" text-anchor="end">50</text><text x="46" y="334" class="ctxc-label mono" text-anchor="end">0</text><text x="22" y="56" class="ctxc-label mono">tokens / K</text>
		<line x1="70" y1="113" x2="950" y2="113" class="ctxc-line-d"></line><line x1="70" y1="119" x2="950" y2="119" class="ctxc-line-w" stroke-dasharray="2 5"></line><text x="100" y="106" class="ctxc-threshold" fill="var(--cs-color-danger)">全量摘要触发线 167K</text><text x="100" y="132" class="ctxc-threshold" fill="var(--cs-color-warning)">读时投影约 90% ≈ 162K</text>
		<polygon class="ctxc-area" points="70,314 107,308 180,255 253,207 261,226 327,174 400,116 437,99 451,135 473,90 510,291 525,259 547,237 620,187 693,135 767,94 781,285 840,242 913,187 950,148 950,330 70,330"></polygon><polyline class="ctxc-line" points="70,314 107,308 180,255 253,207 261,226 327,174 400,116 437,99 451,135 473,90 510,291 525,259 547,237 620,187 693,135 767,94 781,285 840,242 913,187 950,148"></polyline>
		<g><circle cx="70" cy="314" r="3" class="ctxc-point"></circle><circle cx="107" cy="308" r="3" class="ctxc-point"></circle><circle cx="180" cy="255" r="3" class="ctxc-point"></circle><circle cx="253" cy="207" r="3" class="ctxc-point"></circle><circle cx="261" cy="226" r="3" class="ctxc-point"></circle><circle cx="327" cy="174" r="3" class="ctxc-point"></circle><circle cx="400" cy="116" r="3" class="ctxc-point"></circle><circle cx="437" cy="99" r="3" class="ctxc-point"></circle><circle cx="451" cy="135" r="3" class="ctxc-point"></circle><circle cx="473" cy="90" r="3" class="ctxc-point"></circle><circle cx="510" cy="291" r="3" class="ctxc-point"></circle><circle cx="525" cy="259" r="3" class="ctxc-point"></circle><circle cx="547" cy="237" r="3" class="ctxc-point"></circle><circle cx="620" cy="187" r="3" class="ctxc-point"></circle><circle cx="693" cy="135" r="3" class="ctxc-point"></circle><circle cx="767" cy="94" r="3" class="ctxc-point"></circle><circle cx="781" cy="285" r="3" class="ctxc-point"></circle><circle cx="840" cy="242" r="3" class="ctxc-point"></circle><circle cx="913" cy="187" r="3" class="ctxc-point"></circle><circle cx="950" cy="148" r="3" class="ctxc-point"></circle></g>
		<circle cx="261" cy="214" r="13" class="ctxc-m-b"></circle><text x="261" y="218" class="ctxc-m-t" text-anchor="middle">L2</text><circle cx="451" cy="123" r="13" class="ctxc-m-w"></circle><text x="451" y="127" class="ctxc-m-t" text-anchor="middle">L4</text><circle cx="510" cy="279" r="13" class="ctxc-m-d"></circle><text x="510" y="283" class="ctxc-m-t" text-anchor="middle">L5</text><circle cx="781" cy="273" r="13" class="ctxc-m-d"></circle><text x="781" y="277" class="ctxc-m-t" text-anchor="middle">L5</text>
		<text x="70" y="356" class="ctxc-label mono" text-anchor="middle">0</text><text x="180" y="356" class="ctxc-label mono" text-anchor="middle">15</text><text x="261" y="356" class="ctxc-label mono" text-anchor="middle">26</text><text x="327" y="356" class="ctxc-label mono" text-anchor="middle">35</text><text x="400" y="356" class="ctxc-label mono" text-anchor="middle">45</text><text x="451" y="356" class="ctxc-label mono" text-anchor="middle">52</text><text x="510" y="356" class="ctxc-label mono" text-anchor="middle">60</text><text x="620" y="356" class="ctxc-label mono" text-anchor="middle">75</text><text x="693" y="356" class="ctxc-label mono" text-anchor="middle">85</text><text x="781" y="356" class="ctxc-label mono" text-anchor="middle">97</text><text x="913" y="356" class="ctxc-label mono" text-anchor="middle">115</text><text x="950" y="356" class="ctxc-label mono" text-anchor="middle">120</text><text x="510" y="390" class="ctxc-label" text-anchor="middle">会话时间 / min（示意）</text>
	</svg></div>
	<p class="ctxc-note"><strong>读图结论：</strong>最近工作保持高保真，较早历史先被清理或投影，最后才做全量摘要。锯齿高度和频率由工具输出体量、任务阶段、阈值和缓存策略共同决定。</p>
</div>

---

## 2. 大窗口下的压缩约束

更大的窗口只降低压缩频率，不取消上下文管理。模型对长输入的利用能力受证据位置、任务类型和干扰信息影响，名义窗口不代表可用容量。

<div class="ctxc-grid">
	<div class="ctxc-card ctxc-card-d"><h3>召回不稳定</h3><p>证据进入窗口不等于能被稳定使用。证据位置、干扰项和任务阶段都会影响检索与整合，长上下文中的冷门细节尤其容易丢失。</p></div>
	<div class="ctxc-card ctxc-card-w"><h3>成本每轮重复支付</h3><p>Agent 的输入通常远大于输出。旧工具结果若每轮重发，会反复产生 prefill、缓存读写和延迟成本。</p></div>
	<div class="ctxc-card ctxc-card-i"><h3>旧状态污染决策</h3><p>已否决方案、过期报错和失效配置与当前事实同权呈现时，模型可能重复调查、重复执行或基于旧状态行动。</p></div>
</div>

压缩的目标不是最小化 token 数，而是在当前任务阶段保留最能支持下一步决策的高信号集合。

---

## 3. 四种杠杆：按信息损失与恢复成本分工

源手册中的大量策略可以归并为四种杠杆。它们不是互相替代关系，而是按成本和损失程度叠加使用。

<div class="ctxc-table-wrap">
	<table class="ctxc-table">
		<thead><tr><th>杠杆</th><th>移出什么</th><th>恢复方式</th><th>损失与代价</th><th>适用信息</th></tr></thead>
		<tbody>
			<tr><td><strong>工具结果清理</strong><br><span class="ctxc-badge ctxc-badge-s">最轻</span></td><td>旧 <code>tool_result</code> 正文，保留工具调用元数据。</td><td class="ctxc-mono">重新读取、检索或执行</td><td>可能打破清理点之后的 KV Cache。</td><td>大体积、幂等、当前不再引用的观察。</td></tr>
			<tr><td><strong>外置记忆 / 文件</strong><br><span class="ctxc-badge ctxc-badge-s">可逆</span></td><td>把结论、数值、日志或产物写到窗口外。</td><td class="ctxc-mono">路径、URL、索引、query</td><td>一次读写往返和一致性管理。</td><td>精确数值、配置、日志、待办和持久产物。</td></tr>
			<tr><td><strong>结构化摘要</strong><br><span class="ctxc-badge ctxc-badge-w">有损</span></td><td>逐字历史改写成目标、决策、状态和下一步。</td><td class="ctxc-mono">通常只能恢复要点</td><td>需要额外采样；冷门细节可能永久丢失。</td><td>长对话、推理脉络、已结束的决策过程。</td></tr>
			<tr><td><strong>子 Agent 隔离</strong><br><span class="ctxc-badge ctxc-badge-i">架构级</span></td><td>大规模检索或探索过程不进入主窗口，只回传结论。</td><td class="ctxc-mono">重新派发或读取子任务记录</td><td>协调、权限和串行依赖成本。</td><td>相互独立、过程很脏、只需综合结果的子任务。</td></tr>
		</tbody>
	</table>
</div>

这些杠杆的共同判据是：**能否恢复、恢复要付多少成本、丢失后是否会改变世界或破坏任务连续性**。可重取的网页正文适合清理；部署回执、提交哈希和子 Agent 的蒸馏结论不能被当作普通观察处理。

---

## 4. 工具结果：保留调用，替换正文

工具结果不能只按新旧清理。一个结果是否能移出活动窗口，至少取决于四个问题：

- 能否通过路径、查询、URL 或相同参数廉价重建；
- 原操作是否幂等，重新执行会不会改变世界；
- 是否承载一次性状态、副作用凭证或其他 Agent 的蒸馏结论；
- 当前推理链是否仍在引用它。

<div class="ctxc-table-wrap">
	<table class="ctxc-table">
		<thead><tr><th>结果类型</th><th>默认处置</th><th>原因</th><th>保留的恢复句柄</th></tr></thead>
		<tbody>
			<tr><td>文件读取、目录列举、代码搜索</td><td><span class="ctxc-badge ctxc-badge-s">可清理</span></td><td>通常幂等，代码库或磁盘是更稳定的真相源。</td><td>路径、查询条件、读取范围。</td></tr>
			<tr><td>网页检索与抓取</td><td><span class="ctxc-badge ctxc-badge-s">提炼后清理</span></td><td>搜索结果信噪比低，但重新抓取有网络成本。</td><td>URL、查询词、已采纳结论。</td></tr>
			<tr><td>只读 Shell</td><td><span class="ctxc-badge ctxc-badge-s">可清理</span></td><td><code>ls</code>、<code>wc</code>、<code>git status</code> 等可安全重放。</td><td>完整命令和工作目录。</td></tr>
			<tr><td>构建、测试、长日志</td><td><span class="ctxc-badge ctxc-badge-w">收窄或落盘</span></td><td>有效信息集中在退出码、首条错误和相关堆栈。</td><td>日志路径、错误摘要、复跑命令。</td></tr>
			<tr><td>写操作、提交、部署、删除</td><td><span class="ctxc-badge ctxc-badge-d">固定为事实</span></td><td>结果是「操作已发生」的凭证；重放可能造成二次副作用。</td><td>事务号、提交哈希、回执或事实记录。</td></tr>
			<tr><td>子 Agent 结论、后台任务状态</td><td><span class="ctxc-badge ctxc-badge-d">不要裁剪</span></td><td>前者是大量探索后的蒸馏结果，后者丢失会导致重复派发。</td><td>压缩后作为附件重注入。</td></tr>
			<tr><td>记忆工具刚写入的结果</td><td><span class="ctxc-badge ctxc-badge-d">排除清理</span></td><td>刚写的笔记若立即被清，外置记忆没有形成可靠交接。</td><td>笔记路径与写入回执。</td></tr>
		</tbody>
	</table>
</div>

Shell 是最容易误处理的工具。只读探查可以重放；大输出应先落盘再用 <code>grep</code>、<code>head</code>、<code>tail</code> 或聚合查询收窄；写操作必须留下「已完成」的事实；长驻进程只保留句柄、健康状态和日志路径。

清理时不应直接从消息数组中删除元素。多数工具协议要求 <code>tool_use</code> 与 <code>tool_result</code> 成对出现；删除会破坏协议，也会让模型看到「调用过工具但没有结果」的空洞。正确做法是保留消息身份和调用参数，只把结果正文替换成可操作占位符。

<pre class="ctxc-code"><code>{
  "type": "tool_result",
  "tool_use_id": "call_123",
  "content": "[结果已清理：测试通过；完整日志见 /tmp/pytest-123.log]"
}</code></pre>

这个占位符同时给出结论和恢复路径，模型既不会重复跑测试，也知道去哪取细节。裸 <code>[cleared]</code> 缺少这两类信息，不适合作为生产实现。

<div class="ctxc-callout ctxc-callout-w"><strong>可恢复压缩：</strong>只有当 Agent 之后能把信息重建回来时，才允许移除它。网页保留 URL，文件保留路径，查询保留 query；没有恢复通道的一次性状态必须固定或外置。</div>

---

## 5. 分层管线：先不进入，再可恢复清理，最后摘要

生产级实现不应一开始就调用模型总结历史。更合理的是一条成本递增的级联：源头治理最便宜，全量摘要最贵；前一层释放的空间会减少后一层触发次数。

<div class="ctxc-table-wrap">
	<table class="ctxc-table">
		<thead><tr><th>层级</th><th>触发时机</th><th>处理方式</th><th>模型调用</th><th>可逆性</th></tr></thead>
		<tbody>
			<tr><td class="ctxc-mono">L0 源头治理</td><td>工具返回前或 Agent 请求大结果前。</td><td>分页、字段裁剪、聚合查询；超大结果直接写盘并返回指针。</td><td>无</td><td>完全可逆</td></tr>
			<tr><td class="ctxc-mono">L1 即时卸载</td><td>单条结果过大，或模型刚判定材料不再相关。</td><td>正文替换为预览、结论、路径或 URL。</td><td>通常无</td><td>可按句柄重取</td></tr>
			<tr><td class="ctxc-mono">L2 批量清理</td><td>轮次结束、用户回合切换或水位接近阈值。</td><td>按工具元数据、最近窗口、pin 名单和引用关系批量替换旧结果。</td><td>无</td><td>取决于恢复句柄</td></tr>
			<tr><td class="ctxc-mono">L3 读时投影</td><td>组装请求时发现水位较高，但原文仍需保留。</td><td>不修改底层消息，只在请求载荷中隐藏正文或附加局部摘要。</td><td>可能有少量</td><td>可回滚</td></tr>
			<tr><td class="ctxc-mono">L4 全量摘要</td><td>越过自动摘要线、用户手动触发或超长错误后兜底。</td><td>旧历史改写为结构化摘要，并重注入系统提示、最近文件、任务状态和恢复路径。</td><td>一次采样</td><td>细节有损</td></tr>
		</tbody>
	</table>
</div>

工具元数据应描述处置策略，而不是让压缩器硬编码工具名。最小字段包括：是否幂等、是否可重取、典型体积、恢复句柄字段、清前必须抽取的内容，以及 <code>drop_after_read</code>、<code>extract_then_drop</code>、<code>persist</code>、<code>pin</code> 等等级。

<div class="ctxc-grid ctxc-grid-2">
	<div class="ctxc-card ctxc-card-i"><h3>框架做确定性判断</h3><ul><li>统计 token、水位和最近窗口；</li><li>维护 pin、exclude 与工具配对；</li><li>保证落盘成功后才替换正文；</li><li>记录清理位置、释放量和恢复句柄。</li></ul></div>
	<div class="ctxc-card ctxc-card-w"><h3>模型做语义判断</h3><ul><li>判断材料是否仍相关；</li><li>提炼结论并写入外置文件；</li><li>识别方案定稿或任务切换；</li><li>决定哪些冷门数值必须持久化。</li></ul></div>
</div>

主链路可以压缩为一个按水位逐级升级的调度函数：

```python
def sweep(history, water, budget):
    for message in order_by_tier(history):
        if message.pinned or message in last_n(history, budget.recent):
            continue
        if message.tier == "extract_then_drop":
            ok = persist_extract(message)  # 先落盘
            if not ok:
                continue                  # 写失败不清
            message.content = placeholder(message, recovery=path)
        elif message.tier == "drop_after_read" and message.idempotent:
            message.content = placeholder(message, recovery=query)
        if water.freed(history) >= budget.min_sweep:  # 攒够量才打破缓存
            return

    projected = project_payload(history, water)       # 读时投影，不改原文
    if projected.tokens > water.summary_line:
        summary = summarize(history, tools_disabled=True)
        return rebuild(system, summary, files, task_state, transcript_path)
    return projected
```

关键不变量：落盘成功后才替换正文；清理量不足时不打破前缀；摘要请求禁用工具并带递归标记。

---

### 触发时机

成熟实现通常同时使用三类触发：框架阈值负责安全边界，语义时机负责选择好的压缩点，反应式兜底负责处理估算失误。

<div class="ctxc-table-wrap">
	<table class="ctxc-table">
		<thead><tr><th>触发方式</th><th>判据</th><th>适用原因</th><th>边界</th></tr></thead>
		<tbody>
			<tr><td><strong>绝对缓冲</strong></td><td class="ctxc-mono">有效窗口 − 固定输出预算</td><td>摘要输出和下一轮动作所需空间不随窗口线性增长，行为可预测。</td><td>不理解任务是否处在好的语义边界。</td></tr>
			<tr><td><strong>固定百分比</strong></td><td class="ctxc-mono">例如窗口的 50% / 85%</td><td>实现简单，提前避开超长错误。</td><td>大窗口可能过早压缩，小窗口又预留不足。</td></tr>
			<tr><td><strong>模型自主</strong></td><td>研究结论已落盘、方案定稿、用户切换任务、即将读取大量新材料。</td><td>能识别 token 阈值看不到的任务阶段。</td><td>模型可能偏保守，仍需硬阈值兜底。</td></tr>
			<tr><td><strong>反应式兜底</strong></td><td>接口返回超长错误，或单条输出意外撑爆预算。</td><td>保留最近消息与必要状态，极限压缩后重试一次。</td><td>必须有递归守卫和一次性重试限制。</td></tr>
		</tbody>
	</table>
</div>

上述触发判据对应五条依次升高的水位线：工具结果清理线决定何时先清理可重取观察；最近窗口保护当前推理链不被清理；单次最少清理量用于判断释放量是否值得打破前缀缓存；摘要触发线为摘要输出和下一轮动作预留固定预算；硬阻塞线阻止继续发送。各条线的具体取值应来自目标模型的输出分布、计费口径和实测任务，不能照搬产品默认值。

<div class="ctxc-callout ctxc-callout-d"><strong>生产守卫：</strong>压缩请求必须禁用工具、带递归标记、记录独立 token 用量，并在连续失败时熔断。否则压缩失败可能再次触发压缩，形成重试风暴。</div>

---

## 6. 压缩后的上下文重组

全量摘要不是把旧历史替换成一段普通段落。系统提示、最近文件、后台任务、永久指令、工具配置和技能正文各有生命周期，应分通道恢复。

<div class="ctxc-figure">
	<div class="ctxc-figure-head"><h3 class="ctxc-figure-title">压缩后的上下文重组：信息分通道</h3><p class="ctxc-figure-meta">语义、状态、配置和持久指令分开恢复</p></div>
	<div class="ctxc-scroll"><svg class="ctxc-svg" viewBox="0 0 1000 396" role="img" aria-labelledby="channel-title channel-desc">
		<title id="channel-title">压缩前对话经过六条恢复通道组成新消息链</title><desc id="channel-desc">语义信息进入摘要，最近文件和任务状态重新注入，永久指令和操作配置分别重载或重建。</desc>
		<defs><marker id="ctxc-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="ctxc-arrow-head"></path></marker></defs>
		<rect x="14" y="30" width="150" height="300" rx="4" class="ctxc-box" stroke-dasharray="5 5" stroke="var(--cs-color-danger)"></rect>
		<text x="30" y="56" class="ctxc-title" fill="var(--cs-color-danger)">压缩前的对话</text><text x="30" y="76" class="ctxc-label mono">~180K tokens</text>
		<text x="30" y="102" class="ctxc-text">系统提示</text><text x="30" y="124" class="ctxc-text">用户消息与助手回复</text><text x="30" y="146" class="ctxc-text">工具结果与 thinking</text><text x="30" y="168" class="ctxc-text">图像 / 二进制</text><text x="30" y="190" class="ctxc-text">记忆文件内容</text><text x="30" y="212" class="ctxc-text">技能正文</text><text x="30" y="234" class="ctxc-text">后台任务状态</text><text x="30" y="290" class="ctxc-label" fill="var(--cs-color-danger)">旧消息离开活跃 prompt</text><text x="30" y="308" class="ctxc-label" fill="var(--cs-color-danger)">transcript 仍可持久保存</text>
		<rect x="292" y="14" width="330" height="50" rx="4" class="ctxc-box-w"></rect><text x="306" y="33" class="ctxc-text mono" fill="var(--cs-color-warning)">通道 1 · 语义信息 → 摘要</text><text x="306" y="52" class="ctxc-text">意图、概念、文件、报错、方案、待办、下一步</text>
		<rect x="292" y="72" width="330" height="50" rx="4" class="ctxc-box-s"></rect><text x="306" y="91" class="ctxc-text mono" fill="var(--cs-color-success)">通道 2 · 最近文件 → 重新注入</text><text x="306" y="110" class="ctxc-text">按最近修改读取；带单文件与总额预算</text>
		<rect x="292" y="130" width="330" height="50" rx="4" class="ctxc-box-s"></rect><text x="306" y="149" class="ctxc-text mono" fill="var(--cs-color-success)">通道 3 · 程序状态 → 附件</text><text x="306" y="168" class="ctxc-text">后台任务运行、完成、失败状态，防止重复派发</text>
		<rect x="292" y="188" width="330" height="50" rx="4" class="ctxc-box-i"></rect><text x="306" y="207" class="ctxc-text mono" fill="var(--cs-color-info)">通道 4 · 永久指令 → 清缓存重载</text><text x="306" y="226" class="ctxc-text">全局记忆不进摘要，下一轮从磁盘重新读取</text>
		<rect x="292" y="246" width="330" height="50" rx="4" class="ctxc-box-i"></rect><text x="306" y="265" class="ctxc-text mono" fill="var(--cs-color-info)">通道 5 · 操作配置 → 重新构建</text><text x="306" y="284" class="ctxc-text">系统提示不参与压缩；重建工具列表、权限与 MCP</text>
		<rect x="292" y="304" width="330" height="50" rx="4" class="ctxc-box-i"></rect><text x="306" y="323" class="ctxc-text mono" fill="var(--cs-color-info)">通道 6 · 技能正文 → 截断重注</text><text x="306" y="342" class="ctxc-text">设置单项与总额预算，超出时移除最旧内容</text>
		<path d="M164 180 C220 180,232 39,288 39" class="ctxc-arrow"></path><path d="M164 180 C220 180,232 97,288 97" class="ctxc-arrow"></path><path d="M164 180 C220 180,232 155,288 155" class="ctxc-arrow"></path><path d="M164 180 C220 180,232 213,288 213" class="ctxc-arrow"></path><path d="M164 180 C220 180,232 271,288 271" class="ctxc-arrow"></path><path d="M164 180 C220 180,232 329,288 329" class="ctxc-arrow"></path>
		<rect x="700" y="30" width="286" height="300" rx="4" class="ctxc-box" stroke="var(--cs-color-success)"></rect><text x="716" y="56" class="ctxc-title" fill="var(--cs-color-success)">压缩后的新消息链</text><text x="716" y="76" class="ctxc-label mono">~30K tokens</text><text x="716" y="104" class="ctxc-text">① 重建的系统提示</text><text x="716" y="128" class="ctxc-text">② 断点续跑说明</text><text x="716" y="152" class="ctxc-text">③ &lt;summary&gt; 结构化摘要</text><text x="716" y="176" class="ctxc-text">④ 最近文件附件</text><text x="716" y="200" class="ctxc-text">⑤ 后台任务状态附件</text><text x="716" y="224" class="ctxc-text">⑥ 截断后的技能正文</text><text x="716" y="248" class="ctxc-text">⑦ transcript 文件路径</text><text x="716" y="292" class="ctxc-label" fill="var(--cs-color-success)">下一轮自动重载：</text><text x="716" y="310" class="ctxc-label">记忆文件 / 路径域规则</text><path d="M622 180 H696" class="ctxc-arrow"></path>
	</svg></div>
	<p class="ctxc-note"><strong>读图结论：</strong>语义信息进入摘要，易变状态作为附件，永久指令靠重载，操作配置每次重建；摘要不应承担所有职责。</p>
</div>

续跑流程通常是：生成结构化摘要；清理文件状态、路径规则和用户上下文缓存；并发收集最近文件和任务状态；组装系统提示、断点说明、摘要、附件和 transcript 路径；最后替换活动请求上下文。新消息链必须明确告诉模型「这是断点续跑，不是新任务」，自动压缩时还应抑制无意义追问。

摘要内容至少覆盖九类信息：主要请求、关键概念、相关文件、报错与修法、采纳与否决的决策、用户原话、待办、当前断点、下一步动作。用户意图、精确数值和外部副作用不应只依赖摘要转述。

---

## 7. 与 KV Cache 的博弈

压缩能减少后续输入 token，但修改历史会破坏前缀缓存。Agent 每轮通常只在尾部追加，系统提示、工具定义和早期消息高度重复；一旦改动历史中的某个位置，该位置之后的 KV Cache 都可能需要重新计算。

<div class="ctxc-figure">
	<div class="ctxc-figure-head"><h3 class="ctxc-figure-title">前缀失效：改一处，其后全部重算</h3><p class="ctxc-figure-meta">左：改动历史；右：只在尾部追加</p></div>
	<div class="ctxc-scroll"><svg class="ctxc-svg" viewBox="0 0 1060 372" role="img" aria-labelledby="prefix-title prefix-desc">
		<title id="prefix-title">修改历史导致改动点后的缓存全部失效</title><desc id="prefix-desc">左侧修改工具定义后，后续四段都要重算；右侧只追加新动作时，前四段保持命中。</desc>
		<text x="16" y="26" class="ctxc-title" fill="var(--cs-color-danger)">改动历史 → 缓存大面积失效</text><text x="16" y="56" class="ctxc-label mono">step n</text><text x="196" y="56" class="ctxc-label mono">step n+1</text>
		<rect x="16" y="66" width="164" height="34" rx="3" class="ctxc-box"></rect><text x="30" y="88" class="ctxc-text">系统提示</text><rect x="16" y="106" width="164" height="34" rx="3" class="ctxc-box"></rect><text x="30" y="128" class="ctxc-text">工具定义</text><rect x="16" y="146" width="164" height="34" rx="3" class="ctxc-box"></rect><text x="30" y="168" class="ctxc-text">动作 1 / 观察 1</text><rect x="16" y="186" width="164" height="34" rx="3" class="ctxc-box"></rect><text x="30" y="208" class="ctxc-text">动作 2 / 观察 2</text>
		<rect x="196" y="66" width="164" height="34" rx="3" class="ctxc-box-s"></rect><text x="210" y="88" class="ctxc-text">系统提示</text><text x="372" y="88" class="ctxc-label mono" fill="var(--cs-color-success)">命中</text><rect x="196" y="106" width="164" height="34" rx="3" class="ctxc-box-d"></rect><text x="210" y="128" class="ctxc-text">工具定义（已改）</text><rect x="196" y="146" width="164" height="34" rx="3" class="ctxc-box-d"></rect><text x="210" y="168" class="ctxc-text">动作 1 / 观察 1</text><rect x="196" y="186" width="164" height="34" rx="3" class="ctxc-box-d"></rect><text x="210" y="208" class="ctxc-text">动作 2 / 观察 2</text><rect x="196" y="226" width="164" height="34" rx="3" class="ctxc-box-d"></rect><text x="210" y="248" class="ctxc-text">动作 3 / 观察 3</text><text x="372" y="168" class="ctxc-label mono" fill="var(--cs-color-danger)">此后重算</text><text x="196" y="286" class="ctxc-label mono" fill="var(--cs-color-danger)">重算 4 段 · 只命中 1 段</text>
		<line x1="516" y1="40" x2="516" y2="300" stroke="var(--cs-color-border)" stroke-dasharray="4 4"></line>
		<text x="552" y="26" class="ctxc-title" fill="var(--cs-color-success)">只在尾部追加 → 前缀全部命中</text><text x="552" y="56" class="ctxc-label mono">step n</text><text x="732" y="56" class="ctxc-label mono">step n+1</text>
		<rect x="552" y="66" width="164" height="34" rx="3" class="ctxc-box"></rect><text x="566" y="88" class="ctxc-text">系统提示</text><rect x="552" y="106" width="164" height="34" rx="3" class="ctxc-box"></rect><text x="566" y="128" class="ctxc-text">工具定义</text><rect x="552" y="146" width="164" height="34" rx="3" class="ctxc-box"></rect><text x="566" y="168" class="ctxc-text">动作 1 / 观察 1</text><rect x="552" y="186" width="164" height="34" rx="3" class="ctxc-box"></rect><text x="566" y="208" class="ctxc-text">动作 2 / 观察 2</text>
		<rect x="732" y="66" width="164" height="34" rx="3" class="ctxc-box-s"></rect><text x="746" y="88" class="ctxc-text">系统提示</text><rect x="732" y="106" width="164" height="34" rx="3" class="ctxc-box-s"></rect><text x="746" y="128" class="ctxc-text">工具定义</text><rect x="732" y="146" width="164" height="34" rx="3" class="ctxc-box-s"></rect><text x="746" y="168" class="ctxc-text">动作 1 / 观察 1</text><rect x="732" y="186" width="164" height="34" rx="3" class="ctxc-box-s"></rect><text x="746" y="208" class="ctxc-text">动作 2 / 观察 2</text><rect x="732" y="226" width="164" height="34" rx="3" class="ctxc-box-w"></rect><text x="746" y="248" class="ctxc-text">动作 3 / 观察 3</text><text x="908" y="128" class="ctxc-label mono" fill="var(--cs-color-success)">前缀命中</text><text x="908" y="248" class="ctxc-label mono" fill="var(--cs-color-warning)">仅新算此段</text><text x="732" y="286" class="ctxc-label mono" fill="var(--cs-color-success)">重算 1 段 · 前 4 段命中</text>
		<rect x="16" y="310" width="1028" height="48" rx="3" class="ctxc-box"></rect><text x="34" y="332" class="ctxc-text">压缩和清理都是在改动历史：要么攒够释放量再做，要么安排在缓存本来就要失效的时刻。</text><text x="34" y="350" class="ctxc-label">工具集应遮蔽而非删除，历史应 append-only，序列化结果必须确定。</text>
	</svg></div>
</div>

一次清理是否划算不能只看释放了多少 token：

$$
B_{\text{clean}} = C_{\text{saved}} - C_{\text{rewrite}}
$$

$C_{\text{saved}}$ 是后续请求少发送内容的收益，$C_{\text{rewrite}}$ 是清理点之后缓存重新写入的成本。若只释放少量 token，却迫使很长的后缀重算，净收益可能为负。

<div class="ctxc-figure">
	<div class="ctxc-figure-head"><h3 class="ctxc-figure-title">一次清理的净收益</h3><p class="ctxc-figure-meta">绿色为节省，红色为重写，橙色为净收益</p></div>
	<div class="ctxc-scroll"><svg class="ctxc-svg" viewBox="0 0 1000 360" role="img" aria-labelledby="cache-title cache-desc">
		<title id="cache-title">清理释放量较小时净收益可能为负</title><desc id="cache-desc">随着单次清理释放 token 增加，节省输入成本上升；固定缓存重写成本被越过之后，净收益转正。</desc>
		<rect x="620" y="18" width="12" height="12" class="ctxc-bar-s"></rect><text x="640" y="29" class="ctxc-label">省下的输入成本</text><rect x="755" y="18" width="12" height="12" class="ctxc-bar-d"></rect><text x="775" y="29" class="ctxc-label">缓存重写成本</text><line x1="885" y1="24" x2="915" y2="24" class="ctxc-line-w"></line><text x="923" y="29" class="ctxc-label">净收益</text>
		<line x1="70" y1="50" x2="950" y2="50" class="ctxc-gridline"></line><line x1="70" y1="90" x2="950" y2="90" class="ctxc-gridline"></line><line x1="70" y1="130" x2="950" y2="130" class="ctxc-gridline"></line><line x1="70" y1="170" x2="950" y2="170" class="ctxc-gridline"></line><line x1="70" y1="210" x2="950" y2="210" class="ctxc-axis"></line><line x1="70" y1="250" x2="950" y2="250" class="ctxc-gridline"></line>
		<text x="48" y="54" class="ctxc-label mono" text-anchor="end">40</text><text x="48" y="94" class="ctxc-label mono" text-anchor="end">30</text><text x="48" y="134" class="ctxc-label mono" text-anchor="end">20</text><text x="48" y="174" class="ctxc-label mono" text-anchor="end">10</text><text x="48" y="214" class="ctxc-label mono" text-anchor="end">0</text><text x="48" y="254" class="ctxc-label mono" text-anchor="end">-10</text>
		<rect x="112" y="206" width="34" height="4" class="ctxc-bar-s"></rect><rect x="150" y="178" width="34" height="32" class="ctxc-bar-d"></rect><rect x="262" y="198" width="34" height="12" class="ctxc-bar-s"></rect><rect x="300" y="178" width="34" height="32" class="ctxc-bar-d"></rect><rect x="412" y="190" width="34" height="20" class="ctxc-bar-s"></rect><rect x="450" y="178" width="34" height="32" class="ctxc-bar-d"></rect><rect x="562" y="170" width="34" height="40" class="ctxc-bar-s"></rect><rect x="600" y="178" width="34" height="32" class="ctxc-bar-d"></rect><rect x="712" y="130" width="34" height="80" class="ctxc-bar-s"></rect><rect x="750" y="178" width="34" height="32" class="ctxc-bar-d"></rect><rect x="862" y="50" width="34" height="160" class="ctxc-bar-s"></rect><rect x="900" y="178" width="34" height="32" class="ctxc-bar-d"></rect>
		<polyline class="ctxc-line-w" points="131,238 281,230 431,222 581,202 731,162 881,82"></polyline><g fill="var(--cs-color-warning)"><circle cx="131" cy="238" r="4"></circle><circle cx="281" cy="230" r="4"></circle><circle cx="431" cy="222" r="4"></circle><circle cx="581" cy="202" r="4"></circle><circle cx="731" cy="162" r="4"></circle><circle cx="881" cy="82" r="4"></circle></g>
		<text x="131" y="285" class="ctxc-label mono" text-anchor="middle">1K</text><text x="281" y="285" class="ctxc-label mono" text-anchor="middle">3K</text><text x="431" y="285" class="ctxc-label mono" text-anchor="middle">5K</text><text x="581" y="285" class="ctxc-label mono" text-anchor="middle">10K</text><text x="731" y="285" class="ctxc-label mono" text-anchor="middle">20K</text><text x="881" y="285" class="ctxc-label mono" text-anchor="middle">40K</text><text x="510" y="320" class="ctxc-label" text-anchor="middle">单次清理释放的 token（示意）</text>
	</svg></div>
	<p class="ctxc-note"><strong>读图结论：</strong>释放量太小会净亏；达到「单次最少清理量」后才值得打破前缀。缓存接近过期时，重写的边际成本更低。</p>
</div>

缓存友好的纪律是：更正旧事实时追加一条新记录，而不是回改旧观察；禁用工具时在解码或路由层遮蔽，而不是从系统提示中删除工具；序列化字段顺序、时间戳和空白必须确定，避免语义不变却打碎缓存。协议级 <code>cache_edits</code> 可以在服务端缓存层屏蔽旧槽位，但这需要模型服务协议配合，普通客户端无法单独实现。

两种代表性方案利用不同的控制权解决了缓存与压缩的冲突：

<div class="ctxc-grid ctxc-grid-2">
	<div class="ctxc-card ctxc-card-s"><h3>服务端 blob：压缩即尾部项</h3><p>Codex 的服务端压缩把旧历史改写为一个不透明的 compaction blob，作为输入项追加在消息链尾部。由于每轮只在尾部追加，blob 之前的系统提示和工具定义前缀始终命中缓存；再次压缩时旧 blob 被改写为新 blob，体积不随压缩次数累积。代价是 blob 对客户端和模型都不可读，审计与访问控制必须在服务端完成。</p></div>
	<div class="ctxc-card ctxc-card-w"><h3>cache_edits：不改字节，只改可见性</h3><p>Claude Code 在请求中附加编辑指令，告诉服务端「从模型的可见上下文中屏蔽这些旧槽位」。本地 messages 数组保持完整，服务端缓存的原始 token 字节不发生变化，因此前缀缓存不被打碎；被屏蔽的内容在后续请求中仍可恢复。代价是它要求模型服务提供协议级可见性编辑能力，纯应用层客户端无法单独实现。</p></div>
</div>

两者的共同前提是<strong>压缩操作不改变已缓存前缀的字节</strong>：blob 把所有变更集中到尾部新项，cache_edits 只改服务端可见性而不改请求字节。纯客户端的历史重写无法满足这个前提，只能通过攒够清理量或选择缓存过期时机来降低重写成本。

---

## 8. 适用边界、失败模式与生产守卫

压缩策略必须跟随任务阶段。调研阶段的主要负担是可重取的检索正文；编码阶段常是测试和构建输出；排查阶段的 thinking、假设和堆栈更密集；验收阶段则需要精确改动清单。

<div class="ctxc-table-wrap">
	<table class="ctxc-table">
		<thead><tr><th>任务形态</th><th>主要风险</th><th>更合适的策略</th></tr></thead>
		<tbody>
			<tr><td>长对话、反复调整需求</td><td>用户意图和已否决方案被改写或遗忘。</td><td>结构化摘要；用户原话和硬约束逐字保留或外置。</td></tr>
			<tr><td>大规模网页或代码检索</td><td>大量低相关正文污染窗口，检索过程重复。</td><td>子 Agent 隔离、读完即清、保留 URL/query、结论落盘。</td></tr>
			<tr><td>迭代开发和测试</td><td>日志巨大但有效信息稀疏，写操作凭证可能丢失。</td><td>输出落盘，只留退出码和首条错误；提交、部署等事实固定。</td></tr>
			<tr><td>复杂缺陷排查</td><td>假设、堆栈、精确行号经过摘要后快速衰减。</td><td>把已验证结论及时写入问题文件；压缩后按需重读日志和 diff。</td></tr>
			<tr><td>跨文档精确对照</td><td>多份原文必须同时可见，过早清理会造成反复取回。</td><td>保留最近窗口或使用独立工作区；先形成对照表再清理来源。</td></tr>
		</tbody>
	</table>
</div>

### 不适合激进摘要的情况

- 需要精确回忆早期冷门细节，而摘要没有保留恢复句柄；
- 多个变量、配置和数值必须同时保持；
- 多份原文需要直接对照，过早摘要会改变证据关系；
- 安全规则只存在于对话历史，而没有进入系统提示或持久文件；
- 当前推理仍在引用某条工具结果，清理会造成上下文断裂。

### 生产系统必须有的守卫

<div class="ctxc-grid ctxc-grid-2">
	<div class="ctxc-card ctxc-card-d"><h3>失败与递归</h3><ul><li>摘要请求禁用工具，避免工具调用失败导致空摘要；</li><li>给压缩请求打来源标记，压缩过程中不再触发压缩；</li><li>连续失败立即熔断，超长错误只重试一次。</li></ul></div>
	<div class="ctxc-card ctxc-card-w"><h3>事务与审计</h3><ul><li>「提炼后清理」必须先写文件并校验成功；</li><li>记录释放量、清理位置、恢复句柄和失败原因；</li><li>压缩采样本身计入 token 和成本审计。</li></ul></div>
	<div class="ctxc-card ctxc-card-i"><h3>预算与配额</h3><ul><li>最近窗口、单文件、附件总量、重取次数都要有预算；</li><li>单次清理量不足时不打破缓存；</li><li>清理后反复重取时强制落盘或升级策略。</li></ul></div>
	<div class="ctxc-card ctxc-card-s"><h3>保真度验证</h3><ul><li>用真实长轨迹建立回归集；</li><li>分别测试高层事实和冷门细节；</li><li>先最大化召回，再压缩摘要长度。</li></ul></div>
</div>

---

## 9. 实现范式的横向对照

不同实现在「什么时候压、压什么、谁来摘要、如何处理缓存和历史」上有不同取舍。核心分歧可以沿八个维度比较；具体阈值和字段随版本变化，正文只呈现机制差异。

<div class="ctxc-result">
	<div class="ctxc-result-before">
		<div class="ctxc-result-before-head">
			<span class="ctxc-result-before-title">压缩前 · 同一上下文</span>
			<span class="ctxc-result-before-size">约 420K tokens（示意）</span>
		</div>
		<div class="ctxc-win" style="margin-top:0">
			<div class="ctxc-win-block is-keep" style="--rc:var(--cs-color-success);border-color:var(--rc)"><b>用户原话</b>需求与硬约束<span class="tok">6K</span></div>
			<div class="ctxc-win-block is-gone" style="--rc:var(--cs-color-danger);border-color:var(--rc)"><b>检索正文 ×12</b>网页与代码搜索<span class="tok">210K</span></div>
			<div class="ctxc-win-block is-gone" style="--rc:var(--cs-color-danger);border-color:var(--rc)"><b>测试日志 ×5</b>构建输出与堆栈<span class="tok">180K</span></div>
			<div class="ctxc-win-block is-keep" style="--rc:var(--cs-color-info);border-color:var(--rc)"><b>git commit</b>写操作凭证<span class="tok">24K</span></div>
		</div>
	</div>
	<div class="ctxc-result-arrow">↓</div>
	<div class="ctxc-result-grid">
		<section class="ctxc-result-card rc-blob">
			<div class="ctxc-result-card-name">Codex CLI → 12K</div>
			<div class="ctxc-win">
				<div class="ctxc-win-block is-keep"><b>系统提示</b>初始上下文重新注入</div>
				<div class="ctxc-win-block is-keep"><b>用户原话 6K</b>逐字保留</div>
				<div class="ctxc-win-block is-replace"><b>加密 blob</b>含 SHA a3f9b1、方案结论与下一步；检索正文与日志不再可读</div>
				<div class="ctxc-win-block is-gone"><b>检索正文 ×12</b>已收入 blob</div>
				<div class="ctxc-win-block is-gone"><b>测试日志 ×5</b>已收入 blob</div>
			</div>
			<div class="ctxc-win-label">再压缩：旧 blob 进 → 新 blob 出</div>
		</section>
		<section class="ctxc-result-card rc-visible">
			<div class="ctxc-result-card-name">Claude Code</div>
			<div class="ctxc-win">
				<div class="ctxc-win-block is-keep"><b>系统提示与工具定义</b>前缀缓存命中</div>
				<div class="ctxc-win-block is-keep"><b>用户原话与最近消息</b>keep_recent 窗口保留</div>
				<div class="ctxc-win-block is-new"><b>结构化摘要</b>sub-agent 生成，替代旧历史</div>
				<div class="ctxc-win-block is-replace"><b>检索/日志槽位</b>本地完整；服务端 cache_edits 屏蔽</div>
				<div class="ctxc-win-block is-keep"><b>git commit 凭证</b>保留在最近窗口</div>
			</div>
			<div class="ctxc-win-label">旧结果对模型不可见，按句柄可恢复</div>
		</section>
		<section class="ctxc-result-card rc-soft">
			<div class="ctxc-result-card-name">OpenCode → 40K</div>
			<div class="ctxc-win">
				<div class="ctxc-win-block is-keep"><b>最近 40K + 2 回合</b>近期消息与工具结果保留</div>
				<div class="ctxc-win-block is-new"><b>五段摘要</b>目标、决策、待办、文件、下一步</div>
				<div class="ctxc-win-block is-replace"><b>最后一条用户消息</b>从摘要中重放</div>
				<div class="ctxc-win-block is-gone"><b>检索/日志旧消息</b>打 compacted 时间戳，不投影</div>
			</div>
			<div class="ctxc-win-label">原文留在数据库，按时间戳回查</div>
		</section>
		<section class="ctxc-result-card rc-event">
			<div class="ctxc-result-card-name">OpenHands</div>
			<div class="ctxc-win">
				<div class="ctxc-win-block is-keep"><b>投影后的近期事件</b>按 schema 筛选的消息与观察</div>
				<div class="ctxc-win-block is-new"><b>CondensationEvent</b>压缩本身也是一条事件</div>
				<div class="ctxc-win-block is-replace"><b>检索/日志正文</b>读时不投影；EventLog 保留原文</div>
				<div class="ctxc-win-block is-keep"><b>git commit 事件</b>在投影中保留</div>
			</div>
			<div class="ctxc-win-label">压缩器无状态，事件链可完整回放</div>
		</section>
	</div>
</div>

<div class="ctxc-table-wrap">
	<table class="ctxc-table">
		<thead><tr><th>维度</th><th>Codex CLI</th><th>Claude Code</th><th>Gemini CLI</th><th>OpenCode</th></tr></thead>
		<tbody>
			<tr>
				<td><strong>摘要由谁生成</strong></td>
				<td>服务端生成不透明 blob，亦有客户端路径。</td>
				<td>fork 一个 sub-agent，禁用工具、单轮生成。</td>
				<td>客户端生成 XML 状态快照。</td>
				<td>隐藏专用 Agent，输出固定五段结构。</td>
			</tr>
			<tr>
				<td><strong>历史可逆性</strong></td>
				<td>不可逆替换；blob 可跨会话重放。</td>
				<td>本地消息完整，只改服务端可见性。</td>
				<td>不可逆替换。</td>
				<td>软删除，打 compacted 时间戳。</td>
			</tr>
			<tr>
				<td><strong>缓存保护</strong></td>
				<td>blob 恒定在尾部，适配 append-only 前缀。</td>
				<td>协议级 cache_edits，缓存层屏蔽旧槽位。</td>
				<td>无专门机制。</td>
				<td>无专门机制。</td>
			</tr>
			<tr>
				<td><strong>多次压缩</strong></td>
				<td>旧 blob 改写为新 blob，体积不累积。</td>
				<td>层叠，摘要含前次摘要。</td>
				<td>层叠。</td>
				<td>层叠。</td>
			</tr>
			<tr>
				<td><strong>用户原话</strong></td>
				<td>逐字保留（约 64K 预算）。</td>
				<td>纳入摘要。</td>
				<td>纳入摘要。</td>
				<td>纳入摘要，但重放最后一条。</td>
			</tr>
			<tr>
				<td><strong>廉价层</strong></td>
				<td>超长工具输出就地截断。</td>
				<td>五层管线：落盘 → 截断 → Microcompact → Collapse → 摘要。</td>
				<td>summarizeToolOutput，仅 shell 工具。</td>
				<td>Prune：预计释放超过阈值才执行。</td>
			</tr>
			<tr>
				<td><strong>触发策略</strong></td>
				<td>剩余约 5%–15%，随模型窗口变化。</td>
				<td>剩余约 13K，固定缓冲。</td>
				<td>用掉约 50%，百分比缓冲，偏保守。</td>
				<td>剩余约 20K，固定缓冲。</td>
			</tr>
			<tr>
				<td><strong>保留窗口</strong></td>
				<td>最近用户消息，约 20K。</td>
				<td>最近几条工具结果，按条数。</td>
				<td>保留最近约 30%。</td>
				<td>最近约 40K + 2 个用户回合。</td>
			</tr>
		</tbody>
	</table>
</div>

<p style="margin:var(--cs-space-3) 0 0;color:var(--cs-color-text-subtle);font-size:var(--cs-text-3xs)">以上为约数，仅用于比较策略差异：固定缓冲在大窗口下更晚触发，百分比缓冲在大窗口下更早触发；实际数值随模型窗口和产品版本变化。</p>

Deep Agents、LangGraph、Pi 等 Agent 框架不引入新的压缩机制，只是把上述策略编排成工具、图节点或固定缓冲；OpenHands 的事件溯源与 OpenCode 的软删除在历史处理上属于同一范式。真正的分歧仍只有摘要执行位置、历史可逆性和廉价层深度三条。

三条设计分歧决定了其余选择：

- **摘要在哪执行**：服务端可以利用协议做缓存编辑和不透明 blob，但牺牲可读性和审计；客户端摘要可审计、可定制，但无法保护服务端前缀缓存。
- **历史是否保留**：软删除和事件溯源保留回放能力，但需要独立维护存储和投影；不可逆替换简单，但压缩后的信息损失无法撤销。
- **廉价层的深度**：在摘要前做越多确定性处理（落盘、截断、批量清理），全量摘要的触发频率和采样成本越低；只依赖摘要的实现更简单，但每次压缩都更贵。

<div class="ctxc-callout"><strong>控制权决定可选路径：</strong>只控制客户端时，可选外置文件、软删除、读时投影和显式占位符；能控制模型服务协议时，才可能实现 cache_edits 或服务端 blob；两者都可控时，才适合训练专用压缩模型。</div>

---

## 参考资料

- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Context editing](https://docs.anthropic.com/en/docs/build-with-claude/context-editing)
- [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [Explore the context window](https://code.claude.com/docs/en/context-window)
- [Lessons from Building Manus](https://www.manus.im/hi/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)

本页依据用户提供的工程手册重组。具体产品阈值和字段均为特定时期、版本与来源口径；工程实现应以当时官方文档、API 行为和自有任务评测为准。
