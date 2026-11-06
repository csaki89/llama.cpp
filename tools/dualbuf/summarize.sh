#!/usr/bin/env bash
# Aggregate bench-out/results.tsv into a markdown table (mean, min-max over repetitions).
TSV="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/bench-out/results.tsv}"
echo "| mode | prompt | n | prompt tok | prompt t/s mean (min-max) | gen t/s mean (min-max) | output md5 | buffers (MiB) |"
echo "|---|---|---|---|---|---|---|---|"
awk -F'\t' 'NR>1{k=$1"|"$3; if(!(k in o)){o[k]=++c; ord[c]=k}
  n[k]++; ptok[k]=$4; ps[k]+=$5; gs[k]+=$7
  if(!(k in pn)||$5<pn[k])pn[k]=$5; if($5>px[k])px[k]=$5
  if(!(k in gn)||$7<gn[k])gn[k]=$7; if($7>gx[k])gx[k]=$7
  b[k]=$8; if(index(m[k],$9)==0)m[k]=m[k] (m[k]?",":"") $9}
 END{for(i=1;i<=c;i++){k=ord[i]; split(k,a,"|");
  printf "| %s | %s | %d | %s | %.1f (%.1f-%.1f) | %.1f (%.1f-%.1f) | %s | %s |\n",a[1],a[2],n[k],ptok[k],ps[k]/n[k],pn[k],px[k],gs[k]/n[k],gn[k],gx[k],m[k],b[k]}}' "$TSV"
