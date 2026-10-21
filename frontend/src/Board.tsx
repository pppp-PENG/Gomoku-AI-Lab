import {Candidate,coord} from './types';
export function Board({size=9,moves=[],step=moves.length,candidates=[],onMove,disabled=false,heat=false}:{size?:number;moves?:number[];step?:number;candidates?:Candidate[];onMove?:(a:number)=>void;disabled?:boolean;heat?:boolean}){
 const visibleMoves=moves.slice(0,step);
 const board=Array(size*size).fill(0);visibleMoves.forEach((a,i)=>board[a]=i%2===0?1:-1);
 const last=visibleMoves.at(-1); const top=candidates.slice(0,5);
 return <div className="board-wrap"><div className="board" data-to-play={visibleMoves.length%2?'white':'black'} style={{gridTemplateColumns:`repeat(${size},1fr)`}}>
 {board.map((stone,a)=>{const c=candidates.find(c=>c.action===a),rank=top.findIndex(c=>c.action===a);return <button key={a} className={`intersection ${stone===1?'black':stone===-1?'white':''} ${last===a?'last':''}`} disabled={disabled||!!stone||!onMove} onClick={()=>onMove?.(a)} aria-label={`${coord(a,size)}${stone?' 已落子':''}`} style={heat&&c&&!stone?{backgroundColor:`rgba(176,224,128,${.08+c.win*.65})`}:undefined}>
 <span className="grid-cross"/>{stone?<span className="stone"/>:rank>=0?<span className={`rank rank-${rank+1}`}>{rank+1}</span>:<span className="ghost"/>}
 {a<size&&<small className="col-label">{String.fromCharCode(65+a)}</small>}{a%size===0&&<small className="row-label">{Math.floor(a/size)+1}</small>}
 </button>})}</div></div>
}
