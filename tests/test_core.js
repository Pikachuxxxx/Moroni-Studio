const assert=require('node:assert/strict');
const M=require('../app/moroni-core.js');
const f=M.buildCellField({rows:3,cols:4,sides:3,jitter:.6,seed:22});
const fixture=require('./field_fixture.json');
assert.deepEqual(M.buildCellField(fixture.params).centers.map(c=>c.map(v=>+v.toFixed(12))),
                 fixture.centers.map(c=>c.map(v=>+v.toFixed(12))));
for(let i=0;i<f.centers.length;i++)assert(M.pointPolygonSignedDistance(...f.centers[i],f.polys[i])<0);
const square=[];for(let y=0;y<10;y++)for(let x=0;x<10;x++)square.push([x,y]);
const labels=new Int32Array(100);
assert.equal(M.computeMetrics(labels,10,10,1).medianSolidity,1);
assert.equal(M.computeMetrics(labels,10,10,1).disconnectedPct,0);
// A filled rectangle has solidity 1 at any raster size; no hull subsampling.
assert.equal(M.computeMetrics(new Int32Array(10000),100,100,1).medianSolidity,1);
// Square solidity 1 and a three-pixel L with hull area 3.5 have mean median 13/14.
const mixed=new Int32Array([0,0,-1,-1, 0,0,1,1, -1,-1,1,-1]);
assert.equal(M.computeMetrics(mixed,4,3,2).medianSolidity,0.929);
assert(M.exportSVG(f,512,340).includes('<polygon'));
const r=M.renderCPU(f,32,20,{alpha:0,beta:0,gamma:0,useMargin:false});
assert(r.labels.every(x=>x===0));
for(const params of [{rows:1.5},{spacing:NaN},{sides:3.5},{radius:Infinity}])
    assert.throws(()=>M.buildCellField(params));
for(const params of [{alpha:-1},{gamma:NaN},{kappa:1.5},{lambda:-1}])
    assert.throws(()=>M.renderCPU(f,8,8,params));
// Exercise unlit fallback output, including empty background and pixel spacing.
global.ImageData=class {constructor(data,width,height){this.data=data;this.width=width;this.height=height;}};
const maps=M.generatePBRMaps({labels:new Int32Array([0,-1]),sdfs:new Float64Array([-.1,.1]),
    width:2,height:1,bbox:[0,2,0,1],nCells:1,spacing:1},.5);
const color=M.cellColors(1)[0].map(v=>Math.round(255*v));
assert.deepEqual([...maps.diffuse.data.slice(0,3)],color);
assert.deepEqual([...maps.diffuse.data.slice(4,7)],[18,18,18]);
assert.equal(maps.height.data[4],0);
assert.equal(maps.roughness.data[0],64);
assert.equal(maps.roughness.data[4],242);
console.log('Shared generation, raster metric, SVG, ties, validation and unlit fallback tests passed');
