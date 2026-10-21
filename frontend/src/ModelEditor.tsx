import type {Checkpoint} from './types';

export type ModelSpec={architecture:string;size:number;channels:number;blocks:number;dim:number;layers:number;heads:number;mlp_ratio:number};
export const defaultModel=(architecture='cnn'):ModelSpec=>({architecture,size:9,channels:48,blocks:3,dim:64,layers:3,heads:4,mlp_ratio:2});
export function readModel(json:string):ModelSpec{try{return JSON.parse(json)}catch{return defaultModel()}}

export function ModelEditor({label,value,onChange,lockSize=false,sizeLabel,disabled=false,sourceId='',checkpoints=[],onSourceChange}:{label:string;value:ModelSpec;onChange:(v:ModelSpec)=>void;lockSize?:boolean;sizeLabel?:string;disabled?:boolean;sourceId?:string;checkpoints?:Checkpoint[];onSourceChange?:(id:string)=>void}){
 const frozen=disabled||!!sourceId;
 const field=(key:keyof ModelSpec,name:string,min:number,max:number,step=1)=><label>{name}<input aria-label={`${label} ${name}`} type="number" disabled={frozen||(key==='size'&&lockSize)} value={value[key]} min={min} max={max} step={step} onChange={e=>onChange({...value,[key]:Number(e.target.value)})}/></label>;
 return <fieldset className="model-editor" disabled={disabled}><legend>{label}</legend>
 {onSourceChange&&<label>初始权重来源<select aria-label={`${label} 初始权重来源`} value={sourceId} onChange={e=>onSourceChange(e.target.value)}><option value="">随机初始化 · 使用下方架构</option>{sourceId&&!checkpoints.some(c=>c.id===sourceId)&&<option value={sourceId}>源权重当前不可用 · {sourceId.slice(0,8)}</option>}<optgroup label="从已有 Checkpoint 加载权重">{checkpoints.map(c=><option key={c.id} value={c.id}>{c.agent?`原模型 ${c.agent} · `:''}{c.model_config.architecture.toUpperCase()} · {c.model_config.size}×{c.model_config.size} · {c.training_kind==='sft'?`SFT ${c.sft?.completed_epochs} 轮`:`${c.games} 局`} / {c.updates} updates · {c.id.slice(0,8)}</option>)}</optgroup></select></label>}
 {sourceId&&<p className="source-note">{disabled?'已记录原始权重来源；完整恢复读取本任务的网络与训练状态。':'沿用所选模型的结构和权重；新任务的优化器、经验池和进度重新开始。'}</p>}
 <label>架构<select aria-label={`${label} 架构`} disabled={frozen} value={value.architecture} onChange={e=>onChange({...value,architecture:e.target.value})}><option value="cnn">CNN / 残差卷积</option><option value="vit">ViT / Transformer</option></select></label><div className="form-grid">{field('size',sourceId?'棋盘大小（源权重固定）':sizeLabel||(lockSize?'棋盘大小（与另一侧一致）':'棋盘大小'),5,19)}{value.architecture==='cnn'?<>{field('channels','通道数',8,128,4)}{field('blocks','残差块数',1,8)}</>:<>{field('dim','Token 维度',8,128)}{field('layers','Encoder 层数',1,8)}{field('heads','Attention heads',1,8)}{field('mlp_ratio','MLP 倍率',1,4)}</>}</div></fieldset>
}
