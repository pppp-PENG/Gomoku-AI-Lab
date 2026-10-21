import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gomoku_data.adapters import gomoku_party, karesis, gomocup, renjunet
from gomoku_data.core import InvalidGame, canonical_moves, identity, move, replay, sft_samples, split_for, validate
from gomoku_data.pipeline import acquire, build, load_games, sha256, write_json

PROV = dict(source_url="https://example.invalid/synthetic", sha256="0"*64, license_url="synthetic")
SEQUENCE = [112,-127,113,-128,114,-129,115,-130,116]


def game():
    return karesis.parse(SEQUENCE,"synthetic",PROV)


def sgf(xr="FREESTYLE", result="B+R", reason="RESIGNATION", moves=";B[hh];W[ih];B[ii];W[jj];B[hi]", count=5, gn="fixture"):
    return f"(;FF[4]GM[4]SZ[15]CA[UTF-8]CP[CC0-1.0]XGP[gomoku-party-sgf-2]XR[{xr}]RU[{gomoku_party.RULES[xr][1]}]GN[{gn}]RE[{result}]XER[{reason}]XMC[{count}]{moves})"


def synthetic_corpus(root):
    root=Path(root); path=root/"raw/karesis.json"; path.parent.mkdir(parents=True)
    # Distinct ordered games (non-D4-equivalent) with safely separated White moves.
    rows=[[112,-w,113,-(w+1),114,-(w+2),115,-(w+3),116] for w in (1,16,31,46,61,76)]
    path.write_text(json.dumps(rows),encoding="utf8")
    asset=dict(path="raw/karesis.json",url="https://example.invalid/synthetic",sha256=sha256(path),bytes=path.stat().st_size)
    lock=dict(sources={"karesis":dict(status="permitted",license="MIT",license_url="synthetic",files=[asset])})
    config=dict(sources=["karesis"],board_size=15,rulesets=["freestyle"],seed=42,splits=[.5,.25,.25])
    build(root,lock,config,root/"normalized")
    return lock, config


class PipelineTests(unittest.TestCase):
    def test_signed_coordinates_and_replay_result(self):
        g=validate(game()); self.assertEqual(g["winner"],"black")
        self.assertEqual(g["moves"][0],move(0,7,7,"black"))
        self.assertEqual(g["moves"][1],move(1,7,8,"white"))

    def test_unknown_and_incomplete_result_never_last_player_guess(self):
        g=karesis.parse(SEQUENCE[:-1],"x",PROV)
        self.assertIsNone(g["winner"])
        with self.assertRaisesRegex(InvalidGame,"unknown_result"):validate(g)

    def test_malformed_turn_bounds_duplicate_and_postwin(self):
        for seq in ([112,113], [112,-112], [225], [True], [0], SEQUENCE+[-1]):
            with self.subTest(seq=seq), self.assertRaises(InvalidGame):
                validate(karesis.parse(seq,"x",PROV))

    def test_white_win_and_wdl(self):
        seq=[112,-127,113,-128,95,-129,96,-130,80,-131]
        g=validate(karesis.parse(seq,"w",PROV));self.assertEqual(g["winner"],"white")
        g.update(game_hash=identity(g),split="train");samples=list(sft_samples(g))
        self.assertEqual(samples[0]["wdl"],[0,0,1]);self.assertEqual(samples[1]["wdl"],[1,0,0])
        self.assertEqual(samples[1]["history"],[112]);self.assertEqual(samples[1]["target_action"],127)

    def test_standard_overline_differs_from_freestyle(self):
        actions=[0,100,1,102,2,104,4,106,5,108,3]
        moves=[move(i,a%15,a//15,"black" if i%2==0 else "white") for i,a in enumerate(actions)]
        self.assertEqual(replay(moves,ruleset="freestyle"),"black")
        self.assertIsNone(replay(moves,ruleset="standard"))

    def test_sgf_coordinates_and_source_result(self):
        g=validate(gomoku_party.parse(sgf(),PROV))
        self.assertEqual(g["moves"][1],move(1,8,7,"white"))
        self.assertEqual(g["termination"],"resignation")
        self.assertEqual(g["winner"],"black")
        validate(gomoku_party.parse(sgf(result="W+T",reason="TIMEOUT"),PROV))
        validate(gomoku_party.parse(sgf(result="0",reason="DRAW"),PROV))

    def test_sgf_escapes_and_multivalues(self):
        ns=gomoku_party.nodes(r"(;C[a\]b\\c;d]XTC[FISCHER][60][1];B[aa])")
        self.assertEqual(ns[0]["C"],["a]b\\c;d"])
        self.assertEqual(ns[0]["XTC"],["FISCHER","60","1"])

    def test_sgf_fail_closed(self):
        cases=[sgf(xr="RENJU"),sgf(xr="SWAP2"),sgf().replace("SZ[15]","SZ[20]"),
               sgf().replace(";W[ih]",";W[]"),sgf().replace(";W[ih]",";W[pa]"),
               sgf().replace(";W[ih]",";B[ih]"),sgf().replace("XMC[5]","XMC[6]"),
               sgf().replace(";B[hh]",";B[hh]XCP[aa]"),sgf().replace(";B[hh]","AB[hh];B[hh]"),
               sgf().replace(";B[hi])","(;B[hi]))"),sgf().replace("RE[B+R]","RE[W+]"),
               sgf()+"junk",sgf()[:-1],sgf().replace("GN[fixture]","GN[a]GN[b]")]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(InvalidGame):validate(gomoku_party.parse(text,PROV))

    def test_all_d4_variants_same_identity_and_split(self):
        g=game(); key=identity(g)
        for mirror in (False,True):
            for rot in range(4):
                h=copy.deepcopy(g)
                for m in h["moves"]:
                    x,y=m["x"],m["y"]
                    if mirror:x=14-x
                    for _ in range(rot):x,y=14-y,x
                    m.update(x=x,y=y,action=y*15+x)
                self.assertEqual(key,identity(h))
                self.assertEqual(split_for(key),split_for(identity(h)))

    def test_bad_split_fraction(self):
        for fractions in ([.5,.2,.2],[-.1,.5,.6],[float('nan'),0,1]):
            with self.assertRaises(ValueError):split_for("x",fractions=fractions)

    def test_permission_preflight_never_requests_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.urlopen") as request:
            for source in ("renjunet","gomocup"):
                lock={"sources":{source:dict(status="blocked",license="unknown",blocker="permission")}}
                with self.assertRaises(PermissionError):acquire(tmp,lock,[source])
                with self.assertRaises(PermissionError):list(renjunet.records(None,None)) if source=="renjunet" else gomocup.records(None,None)
            request.assert_not_called()

    def test_provisional_psq_parser_requires_base_size_and_metadata(self):
        text="Piskvorky 15x15, 5:5, 0\n8,8,0\n8,9,2\n-\nNames"
        g=gomocup.parse_psq(text,coordinate_base=1,ruleset="freestyle",game_id="fixture",provenance=PROV,winner="black",termination="resignation")
        self.assertEqual(g["moves"][0]["action"],112)
        with self.assertRaises(InvalidGame):gomocup.parse_psq(text.replace("15x15","20x20"),coordinate_base=1,ruleset="freestyle",game_id="x",provenance=PROV,winner="black")

    def test_reproducible_build_whole_game_splits_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lock,config=synthetic_corpus(root)
            a=json.loads((root/"normalized/manifest.json").read_text())
            b=build(root,lock,config,root/"second")
            self.assertEqual(a,b)
            games=list(load_games(root/"normalized",["karesis"]))
            self.assertEqual(len(games),6)
            sets=[{g["game_hash"] for g in load_games(root/"normalized",["karesis"],s)} for s in ("train","validation","test")]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            with self.assertRaises(ValueError):build(root,lock,config,root/"normalized")
            p=root/"normalized/games.jsonl";p.write_text(p.read_text()+"\n")
            with self.assertRaises(ValueError):list(load_games(root/"normalized",["karesis"]))
            raw=root/"raw/karesis.json";raw.write_text("[]")
            with self.assertRaises(ValueError):acquire(root,lock,["karesis"],offline=True)

    def test_duplicates_quarantine_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/"raw.json";path.write_text(json.dumps([SEQUENCE,SEQUENCE,[0]]))
            lock=dict(sources={"karesis":dict(status="permitted",license="MIT",license_url="synthetic",files=[dict(path="raw.json",url="https://example.invalid",sha256=sha256(path),bytes=path.stat().st_size)])})
            cfg=dict(sources=["karesis"],board_size=15,rulesets=["freestyle"])
            result=build(root,lock,cfg,root/"out")
            self.assertEqual((result["unique_games"],result["counts"]["karesis.duplicate"],result["quarantine"]),(1,1,1))
            # Same placements with conflicting source results: exclude BOTH.
            a=gomoku_party.parse(sgf(),PROV); b=gomoku_party.parse(sgf(result="W+R",gn="other"),PROV)
            asset=root/"party.gz";asset.write_bytes(gzip.compress((sgf()+"\n"+sgf(result="W+R",gn="other")).encode()))
            lock=dict(sources={"gomoku_party":dict(status="permitted",license="CC0-1.0",license_url="synthetic",files=[dict(path="party.gz",url="https://example.invalid",sha256=sha256(asset),bytes=asset.stat().st_size)])})
            cfg["sources"]=["gomoku_party"];r=build(root,lock,cfg,root/"conflict")
            self.assertEqual(r["unique_games"],0);self.assertEqual(r["quarantine"],2)

    def test_cross_source_alias_selection_keeps_same_game_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            k=root/'k.json';k.write_text(json.dumps([SEQUENCE]))
            coords=''.join(';'+('B' if v>0 else 'W')+'['+chr(97+abs(v)%15)+chr(97+abs(v)//15)+']' for v in SEQUENCE)
            text=sgf(result='B+',reason='FIVE_IN_ROW',moves=coords,count=9)
            p=root/'p.gz';p.write_bytes(gzip.compress(text.encode()))
            sources={}
            for name,path,license in [('karesis',k,'MIT'),('gomoku_party',p,'CC0-1.0')]:
                sources[name]=dict(status='permitted',license=license,license_url='synthetic',files=[dict(path=path.name,url='https://example.invalid',sha256=sha256(path),bytes=path.stat().st_size)])
            config=dict(sources=list(sources),board_size=15,rulesets=['freestyle'])
            result=build(root,dict(sources=sources),config,root/'out')
            self.assertEqual(result['unique_games'],1)
            a=list(load_games(root/'out',['karesis']));b=list(load_games(root/'out',['gomoku_party']))
            self.assertEqual(a,b);self.assertEqual(len(a[0]['aliases']),2)


if __name__ == "__main__":unittest.main()
