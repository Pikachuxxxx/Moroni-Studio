/* Finite-field WebGL2 renderer. The CPU-generated polygon buffer is authoritative.
 * Complete candidate scan with conservative rejection; no fixed spatial stencil.
 * GPU timer queries are optional. Host submission time is never called GPU time.
 */
const VERTEX_SHADER_SRC=`#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main(){v_uv=(a_position+1.)*.5;gl_Position=vec4(a_position,0.,1.);}`;
const FRAGMENT_SHADER_SRC=`#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_cells;
uniform int u_count,u_mode,u_samples;
uniform vec2 u_resolution;
uniform vec4 u_bbox;
uniform float u_alpha,u_beta,u_gamma,u_kappa,u_lambda,u_margin,u_radius,u_spacing,u_split_pos;
uniform bool u_use_margin,u_show_wireframe,u_prune;
uniform vec3 u_grout_color;
in vec2 v_uv;
out vec4 fragColor;
struct Result {int id;float energy;float sc;float dc;float phase;bool grout;};
float sdPolygon(vec2 p,int c,int n){
    float d=1e20;bool inside=false;
    vec2 a=texelFetch(u_cells,ivec2(n,c),0).xy;
    for(int k=0;k<12;k++){
        if(k>=n)break;
        vec2 b=texelFetch(u_cells,ivec2(k+1,c),0).xy;
        vec2 e=b-a,w=p-a;
        float t=clamp(dot(w,e)/max(dot(e,e),1e-30),0.,1.);
        d=min(d,length(w-t*e));
        if((a.y>p.y)!=(b.y>p.y))
            if(p.x<a.x+(p.y-a.y)*e.x/e.y)inside=!inside;
        a=b;
    }
    return inside ? -d:d;
}
Result evaluate(vec2 p,bool vor){
    Result best=Result(0,1e30,0.,0.,0.,false);
    for(int c=0;c<256;c++){
        if(c>=u_count)break;
        vec4 cell=texelFetch(u_cells,ivec2(0,c),0);
        vec2 delta=p-cell.xy;float dc=length(delta);
        float lower=vor ? dc:(u_alpha+u_beta)*dc-u_beta*cell.z;
        // Slack protects near ties from small rounding errors; validation reports
        // float32 disagreement separately from the real-arithmetic bound.
        if(u_prune && lower>best.energy+1e-5*(1.+abs(best.energy)))continue;
        float sc=sdPolygon(delta,c,int(cell.w));
        float ph=dc<1e-6 ? 0.:1.-cos(u_kappa*atan(delta.y,delta.x)+u_lambda*dc);
        float e=vor ? dc:u_alpha*dc+u_beta*sc+u_gamma*ph;
        if(e<best.energy)best=Result(c,e,sc,dc,ph,false);
    }
    best.grout=u_use_margin && best.sc>u_margin;
    return best;
}
float heightAt(Result e){return e.grout ? 0.:sqrt(clamp(-e.sc/u_radius,0.,1.));}
vec3 palette(int id){
    return texelFetch(u_cells,ivec2(13,id),0).rgb;
}
vec3 shade(vec2 p,bool vor){
    Result e=evaluate(p,vor);
    if(u_mode==8){int id=e.grout ? 0:e.id+1;return vec3(float(id%256),float(id/256),0.)/255.;}
    if(u_mode==4)return vec3(clamp(e.dc/u_spacing,0.,1.));
    if(u_mode==5)return mix(vec3(1.),e.sc<0.?vec3(.1,.3,.9):vec3(.9,.2,.1),clamp(abs(e.sc)/u_radius,0.,1.));
    if(u_mode==6)return vec3(e.phase*.5);
    if(u_mode==2)return vec3(heightAt(e));
    if(u_mode==3)return vec3(e.grout ? .95:.25);
    if(u_mode==1){
        float eps=.005*u_spacing;
        float dx=(heightAt(evaluate(p+vec2(eps,0.),vor))-heightAt(evaluate(p-vec2(eps,0.),vor)))/(2.*eps);
        float dy=(heightAt(evaluate(p+vec2(0.,eps),vor))-heightAt(evaluate(p-vec2(0.,eps),vor)))/(2.*eps);
        // World-height scale .06*spacing; +Y is downward in the exported image.
        return normalize(vec3(-dx*.06*u_spacing,-dy*.06*u_spacing,1.))*.5+.5;
    }
    if(u_show_wireframe && abs(e.sc)<.008*u_spacing)return vec3(.1);
    return e.grout ? vec3(.07):palette(e.id); // Shared unlit paper palette.
}
void main(){
    vec2 uv=vec2(v_uv.x,1.-v_uv.y);
    vec2 span=vec2(u_bbox.y-u_bbox.x,u_bbox.w-u_bbox.z);
    vec2 p=vec2(u_bbox.x,u_bbox.z)+uv*span;
    bool vor=u_mode==7 && uv.x<u_split_pos;
    vec3 c;
    if(u_samples==4 && u_mode!=8){
        vec2 d=span/u_resolution*.25;
        c=.25*(shade(p+d,vor)+shade(p-d,vor)+shade(p+vec2(d.x,-d.y),vor)+shade(p+vec2(-d.x,d.y),vor));
    }else c=shade(p,vor);
    if(u_mode==7 && abs(uv.x-u_split_pos)<.002)c=vec3(1.);
    fragColor=vec4(c,1.);
}`;

class MoroniGLRenderer {
    constructor(canvas){
        this.canvas=canvas;
        this.gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});
        if(!this.gl)throw new Error('WebGL2 unavailable');
        const gl=this.gl;
        const vs=this.compile(gl.VERTEX_SHADER,VERTEX_SHADER_SRC),fs=this.compile(gl.FRAGMENT_SHADER,FRAGMENT_SHADER_SRC);
        this.program=gl.createProgram();gl.attachShader(this.program,vs);gl.attachShader(this.program,fs);gl.linkProgram(this.program);
        if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));
        this.vao=gl.createVertexArray();gl.bindVertexArray(this.vao);
        const vbo=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vbo);
        gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
        const loc=gl.getAttribLocation(this.program,'a_position');gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
        this.texture=gl.createTexture();this.uniforms={};this.pending=[];this.gpuMs=null;
        this.timer=gl.getExtension('EXT_disjoint_timer_query_webgl2');
        this.timingSamples=[];this.disjointSamples=0;
    }
    compile(type,source){const gl=this.gl,s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);
        if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
    uniform(name){return this.uniforms[name]??(this.uniforms[name]=this.gl.getUniformLocation(this.program,name));}
    setField(params){
        const key=JSON.stringify(['rows','cols','spacing','radius','jitter','sides','seed'].map(k=>params[k]));
        if(key===this.fieldKey)return;
        this.fieldKey=key;this.field=MoroniCore.buildCellField(params);
        const f=this.field,data=new Float32Array(14*f.centers.length*4),colors=MoroniCore.cellColors(f.centers.length);
        f.centers.forEach((c,i)=>{
            data.set([c[0],c[1],f.radii[i]+1e-6,f.polys[i].length],i*56);
            f.polys[i].forEach((v,j)=>data.set([v[0]-c[0],v[1]-c[1],0,0],i*56+4*(j+1)));
            data.set([...colors[i],1],i*56+52);
        });
        const gl=this.gl;gl.bindTexture(gl.TEXTURE_2D,this.texture);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,14,f.centers.length,0,gl.RGBA,gl.FLOAT,data);
    }
    pollTiming(){
        const gl=this.gl;if(!this.timer)return;
        if(gl.getParameter(this.timer.GPU_DISJOINT_EXT)){
            this.disjointSamples+=this.pending.length;this.pending.forEach(q=>gl.deleteQuery(q));this.pending=[];this.gpuMs=null;return;
        }
        while(this.pending.length && gl.getQueryParameter(this.pending[0],gl.QUERY_RESULT_AVAILABLE)){
            const q=this.pending.shift();this.gpuMs=gl.getQueryParameter(q,gl.QUERY_RESULT)/1e6;
            this.timingSamples.push(this.gpuMs);if(this.timingSamples.length>1000)this.timingSamples.shift();gl.deleteQuery(q);
        }
    }
    render(p,time=0,options={}){
        const gl=this.gl;this.setField(p);this.pollTiming();
        const width=options.width??this.canvas.width,height=options.height??this.canvas.height;
        gl.bindFramebuffer(gl.FRAMEBUFFER,options.framebuffer??null);gl.viewport(0,0,width,height);
        gl.useProgram(this.program);gl.bindVertexArray(this.vao);gl.disable(gl.DITHER);
        gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.texture);gl.uniform1i(this.uniform('u_cells'),0);
        gl.uniform1i(this.uniform('u_count'),this.field.centers.length);gl.uniform4fv(this.uniform('u_bbox'),this.field.bbox);
        gl.uniform2f(this.uniform('u_resolution'),width,height);
        const defaults={alpha:.58,beta:.32,gamma:.1,kappa:2,lambda:1,margin:.03,radius:.5,spacing:.9,split_pos:.5};
        for(const [k,v] of Object.entries(defaults))gl.uniform1f(this.uniform('u_'+k),p[k==='split_pos'?'splitPos':k]??v);
        gl.uniform1i(this.uniform('u_mode'),p.mode??0);gl.uniform1i(this.uniform('u_samples'),p.samples??1);
        gl.uniform1i(this.uniform('u_use_margin'),p.useMargin??true);gl.uniform1i(this.uniform('u_show_wireframe'),p.showWireframe??false);
        gl.uniform1i(this.uniform('u_prune'),p.prune??true);gl.uniform3fv(this.uniform('u_grout_color'),p.groutColor??[.12,.12,.12]);
        let q=null;
        if(this.timer && options.timing!==false && this.pending.length<8){q=gl.createQuery();gl.beginQuery(this.timer.TIME_ELAPSED_EXT,q);}
        gl.drawArrays(gl.TRIANGLES,0,6);
        if(q){gl.endQuery(this.timer.TIME_ELAPSED_EXT);this.pending.push(q);}
    }
    readLabels(params,width=180,height=120){
        const gl=this.gl,fb=gl.createFramebuffer(),tex=gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D,tex);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,width,height,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
        gl.bindFramebuffer(gl.FRAMEBUFFER,fb);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
        if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error('Readback framebuffer incomplete');
        this.render({...params,mode:8,samples:1},0,{width,height,framebuffer:fb,timing:false});
        const pixels=new Uint8Array(width*height*4),labels=new Int32Array(width*height);
        gl.readPixels(0,0,width,height,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
        for(let y=0;y<height;y++)for(let x=0;x<width;x++){
            const j=((height-1-y)*width+x)*4;labels[y*width+x]=pixels[j]+256*pixels[j+1]-1;
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.deleteFramebuffer(fb);gl.deleteTexture(tex);
        return {labels,width,height};
    }
}
window.MoroniGLRenderer=MoroniGLRenderer;
