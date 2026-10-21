export type Checkpoint = {id:string; experiment:string; created:number; games:number; updates:number; parameters:number; estimated_flops:number; model_config: Record<string,any>; config:Record<string,any>; agent?:'A'|'B'|null;pair_id?:string|null;training_kind?:string;sft?:{completed_epochs:number;train_samples:number;validation_samples:number}};
export type Game = {id:string;created:number;size:number;moves:number[];board:number[][];human:number;to_play:number;winner:number|null;terminal:boolean;checkpoint:string;simulations:number;revision:number;parent?:{game:string;step:number}};
export type Candidate = {action:number;win:number;draw:number;loss:number;value:number;policy:number;visit_fraction:number;visits:number;search_q:number|null;search_wdl:number[]|null};
export type Analysis = {candidates:Candidate[];top5:Candidate[];root_wdl:number[];elapsed_ms:number;simulations:number};
export const coord=(a:number,n:number)=>`${String.fromCharCode(65+a%n)}${Math.floor(a/n)+1}`;
export const pct=(p:number)=>`${(p*100).toFixed(1)}%`;
export async function api<T=any>(path:string,body?:unknown,method=body===undefined?'GET':'POST'):Promise<T>{
  const r=await fetch('/api'+path,{method,...(body===undefined?{}:{headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});
  if(!r.ok){const e=await r.json().catch(()=>({detail:r.statusText})); throw new Error(typeof e.detail==='string'?e.detail:JSON.stringify(e.detail));}return r.json();
}
