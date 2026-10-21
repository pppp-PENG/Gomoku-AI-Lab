import {useEffect,useState} from 'react';
import {Check,Download,RotateCcw,Save,ChevronLeft,ChevronRight} from 'lucide-react';
import {Board} from './Board';
import {api,coord} from './types';

export const collectionSizes=[9,13,5,15,19];
export type Demonstration={id:string;created:number;name:string;size:number;moves:number[];winner:number;rules:string};
type Draft={size:number;moves:number[];name:string};
type Position={terminal:boolean;winner:number|null;to_play:number};
const key='gomoku-human-draft-v1';
const approvalKey='gomoku-human-draft-approved-v1';
function readDraft():Draft{try{const d=JSON.parse(localStorage.getItem(key)||'null');if(d&&Number.isInteger(d.size)&&Array.isArray(d.moves)&&typeof d.name==='string')return d;}catch{}return {size:9,moves:[],name:''};}
function readApproval(){try{const a=JSON.parse(localStorage.getItem(approvalKey)||'null'),d=readDraft();if(a&&a.size===d.size&&JSON.stringify(a.moves)===JSON.stringify(d.moves)&&typeof a.id==='string')return a.id;}catch{}return '';}
export const resultName=(winner:number|null)=>winner===0?'平局':winner===1?'黑棋获胜':winner===-1?'白棋获胜':'进行中';

export function DataCollection({onSFT}:{onSFT:(size:number)=>void}){
 const[draft,setDraft]=useState<Draft>(readDraft),[position,setPosition]=useState<Position|null>(null),[data,setData]=useState<Demonstration[]>([]),[busy,setBusy]=useState(false),[error,setError]=useState(''),[saved,setSaved]=useState(readApproval),[review,setReview]=useState<Demonstration|null>(null),[step,setStep]=useState(0);
 const[archiveSize,setArchiveSize]=useState(()=>readDraft().size);
 const archive=data.filter(g=>g.size===archiveSize);
 const reload=async()=>setData(await api('/demonstrations'));
 useEffect(()=>{reload().catch(e=>setError(String(e)));api<Position>('/demonstrations/position',readDraft()).then(setPosition).catch(e=>setError(String(e)))},[]);
 useEffect(()=>{localStorage.setItem(key,JSON.stringify(draft))},[draft]);
 async function run(fn:()=>Promise<void>){setBusy(true);setError('');try{await fn()}catch(e){setError(e instanceof Error?e.message:String(e))}finally{setBusy(false)}}
 async function change(moves:number[],size=draft.size){const next={...draft,size,moves};const p=await api<Position>('/demonstrations/position',next);setDraft(next);setPosition(p);if(size!==draft.size)setArchiveSize(size);setSaved('');localStorage.removeItem(approvalKey);setReview(null)}
 async function save(){const record=await api<Demonstration>('/demonstrations',draft);setSaved(record.id);localStorage.setItem(approvalKey,JSON.stringify({id:record.id,size:draft.size,moves:draft.moves}));await reload()}
 const current=review||draft;
 const shown=review?step:draft.moves.length;
 return <>
 {error&&<div className="alert error" role="alert">{error}</div>}
 <div className="stats"><div className="stat"><span>已收录棋谱</span><strong>{data.length}<small>局</small></strong><p>仅包含完成后明确保存的棋局</p></div><div className="stat"><span>可用落子示范</span><strong>{data.reduce((n,g)=>n+g.moves.length,0)}<small>手</small></strong><p>黑棋与白棋每一步均参与 SFT</p></div></div>
 <div className="game-layout"><section className="panel board-panel"><div className="section-heading"><div><h2>{review?'已收录棋谱复看':'自己执黑白，记录一局棋'}</h2><p>{review?`${review.name||review.id.slice(0,8)} · ${resultName(review.winner)}`:position?.terminal?resultName(position.winner):`轮到${draft.moves.length%2?'白':'黑'}棋落子`}</p></div><span className="badge">{shown} 手 · {current.size} × {current.size}</span></div>
 <Board size={current.size} moves={current.moves} step={shown} disabled={busy||!!review||!position||position.terminal||!!saved} onMove={a=>run(()=>change([...draft.moves,a]))}/>
 {review?<div className="timeline"><label>采集棋谱时间轴<input aria-label="采集棋谱时间轴" type="range" min={0} max={review.moves.length} value={step} onChange={e=>setStep(+e.target.value)}/></label><div className="inline-actions"><button onClick={()=>setStep(Math.max(0,step-1))}><ChevronLeft size={15}/>上一步</button><button onClick={()=>setStep(Math.min(review.moves.length,step+1))}>下一步<ChevronRight size={15}/></button><button onClick={()=>setReview(null)}>返回采集草稿</button></div></div>:<><div className="board-footer"><span><i className={`player-dot ${draft.moves.length%2?'white':''}`}/>{position?.terminal?resultName(position.winner):'黑白自动交替，无 AI 代下'}</span><span>草稿保存在此浏览器</span></div><div className="inline-actions"><button disabled={busy||!draft.moves.length||!!saved} onClick={()=>run(()=>change(draft.moves.slice(0,-1)))}><RotateCcw size={14}/>撤销一手</button><button disabled={busy} onClick={()=>run(()=>change([]))}>{saved?'已收录 · 开始下一局':position?.terminal?'不收录，开始下一局':'重新开始'}</button></div></>}
 <div className="move-list">{current.moves.slice(0,shown).map((a,i)=><span key={i}><small>{i+1}.</small><i className={`player-dot ${i%2?'white':''}`}/>{coord(a,current.size)}</span>)}</div></section>
 <div><section className="panel"><div className="section-heading"><h2>收录设置</h2><Save size={17}/></div><label>采集棋盘大小<select aria-label="采集棋盘大小" value={draft.size} disabled={busy||draft.moves.length>0} onChange={e=>run(()=>change([],+e.target.value))}>{collectionSizes.map(n=><option key={n} value={n}>{n} × {n}</option>)}</select></label><label>棋谱名称（可选）<input aria-label="棋谱名称" maxLength={120} value={draft.name} disabled={!!saved} onChange={e=>setDraft({...draft,name:e.target.value})} placeholder="例如：进攻与防守练习"/></label><p>按黑、白顺序直接落子。五连或棋盘填满后，你可以决定是否将整局棋谱加入训练数据。</p>
 {saved?<div className="note green"><Check size={14}/> 已收录 · {saved.slice(0,8)}。黑白双方全部落子已保存。</div>:<button className="primary full" disabled={busy||!position?.terminal||!!review} onClick={()=>run(save)}><Save size={14}/>{busy?'正在处理…':'收录这局棋谱'}</button>}
 <p className="note">未点击收录的草稿不会进入训练集。已收录棋谱保持原样；相同落子序列重复提交不会重复计数。</p><button className="full" onClick={()=>onSFT(draft.size)}>前往 SFT 训练<ChevronRight size={14}/></button></section>
 <section className="panel"><div className="section-heading"><h2>已收录数据</h2><a className="button" href="/api/demonstrations/export" download><Download size={13}/>导出</a></div><label>已收录棋谱分类<select aria-label="已收录棋谱分类" value={archiveSize} onChange={e=>setArchiveSize(+e.target.value)}>{[...new Set([...collectionSizes,...data.map(g=>g.size)])].map(n=><option key={n} value={n}>{n} × {n} · {data.filter(g=>g.size===n).length} 局</option>)}</select></label><div className="archive">{archive.map(g=><button key={g.id} className={review?.id===g.id?'active':''} onClick={()=>{setReview(g);setStep(g.moves.length)}}><span>{g.name||g.id.slice(0,8)}<small>{g.size}×{g.size} · {new Date(g.created*1000).toLocaleString()}</small></span><span>{g.moves.length} 手<small>{resultName(g.winner)}</small></span></button>)}</div>{!archive.length&&<p>暂无 {archiveSize} × {archiveSize} 棋谱。完成该尺寸棋局后选择收录。</p>}</section></div></div></>;
}
