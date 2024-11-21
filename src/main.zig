const std = @import("std");
fn MulDim(comptime A: type, comptime B: type) type {
    comptime std.debug.assert(A.Child == B.Child);
    comptime if (A.columns != B.rows) {
        @compileError("MxN matrices can only be multiplied with NxP matrices");
    };

    return Matrix(A.Child, A.rows, B.columns);
}
fn TranspDim(comptime A: type) type {
    return Matrix(A.Child, A.columns, A.rows);
}


fn PairMatrix(comptime A: type, comptime B: type) type {
    return Matrix(struct{a:A.Child, b:B.Child}, A.rows, B.columns);
}

pub fn Matrix(
    comptime T:type,
    comptime m: usize,
    comptime n: usize,
    ) type {
    return struct {
        const Self = @This();
        pub const Child=T;
        pub const rows=m;
        pub const columns=n;
        data: [rows*columns]T,
        pub fn init() Self {
            return Self {
                .data = [_]T{undefined} ** (rows * columns),
            };
        }

        pub fn from_values(data: [rows * columns]T) Self {
            var out = Self.init();
            std.mem.copyForwards(T, &out.data, &data);
            return out;
        }
        pub fn fill(_:Self,value:T) Self{
            var out=Self.init();
            for (0..rows) |i| {
                for (0..columns) |j| {
                    out.data[i*columns+j]=value;
                }
            }
            return out;

        }
        pub fn print(s: Self) Self {
            std.debug.print("\n", .{});
            for (0..rows) |i| {
                if(i==0){
                    std.debug.print("/", .{});
                }
                else if(i==rows-1){
                    std.debug.print("\\", .{});
                }
                else{
                    std.debug.print("|",.{});
                }
                for (0..columns) |j| {
                    if(j==columns-1){
                        std.debug.print("{any: ^7.1}",.{s.data[columns*i+j]});
                    } else {
                        std.debug.print("{any: ^7.1} ",.{s.data[columns*i+j]});
                    }
                }
                 if(i==0){
                    std.debug.print("\\\n", .{});
                }
                else if(i==rows-1){
                    std.debug.print("/\n", .{});
                }
                else{
                    std.debug.print("|\n",.{});
                }
            }
            return s; 
        }
        pub fn set(s: *Self, i: usize, j: usize, value: T) void {
            std.debug.assert(i<rows);
            std.debug.assert(j<columns);
            s.data[i*columns+j]=value;
        }
        pub fn MatMul(a: Self, b: anytype) MulDim(@TypeOf(a), @TypeOf(b)) {
            comptime var Out=MulDim(@TypeOf(a), @TypeOf(b));
            var out=Out.init();
            for (0..Out.rows) |i| {
                for (0..Out.columns) |j| {
                    out.data[i*Out.columns+j]=0;
                    for (0..columns) |k| {
                        out.data[i*Out.columns+j]+=a.data[i*Self.columns+k]*b.data[k*@TypeOf(b).columns+j];
                    }
                }
            }
            return out;
        }
        pub fn Transpose(A:Self)TranspDim(@TypeOf(A)){
            comptime var Out=TranspDim(@TypeOf(A));
            var out=Out.init();
            for (0..Self.rows) |i| {
                for (0..Self.columns) |j| {
                    out.data[j*Out.columns+i]=A.data[i*Self.columns+j];
                }
            }
            return out;
        }
        pub fn add(A:Self,B:Self) Self {
            var out=Self.init();
            for (0..rows) |i| {
                for (0..columns) |j| {
                    out.data[i*columns+j]=A.data[i*columns+j]+B.data[i*columns+j];
                }
            }
            return out;
        }
        pub fn sub(A:Self,B:Self) Self {
            var out=Self.init();
            for (0..rows) |i| {
                for (0..columns) |j| {
                    out.data[i*columns+j]=A.data[i*columns+j]-B.data[i*columns+j];
                }
            }
            return out;
        }
        pub fn ScalarMul(A:Self,B:T) Self {
            var out=Self.init().fill(0);
            for (0..rows) |i| {
                for (0..columns) |j| {
                    out.data[i*columns+j]=A.data[i*columns+j]*B;
                }
            }
            return out;
        }
        pub fn ApplyFunc(A:Self,comptime f:(fn(x:T) T)) Self {
            var out=Self.init();
            for (0..rows) |i| {
                for (0..columns) |j| {
                    out.data[i*columns+j]=f(A.data[i*columns+j]);
                }
            }
            return out;
        }
        pub fn getIJ(s:Self, i:usize, j:usize) T {
            return s.data[i*columns+j];
        }
        pub fn pairWise(A:Self, B:anytype) PairMatrix(@TypeOf(A), @TypeOf(B)){
            comptime var Out=PairMatrix(@TypeOf(A), @TypeOf(B));
            var out = Out.init();
            for (0..rows) |i| {
                for (0..columns) |j| {
                    out.data[i*columns+j].a=A.data[i*columns+j];
                    out.data[i*columns+j].b=B.data[i*columns+j];
                }
            }
            return out;
        }

    };
}
pub fn HiddenLayer(
    comptime nodes:usize,
    comptime inputNodes: usize,
    ) type {
    return struct{
        const Self=@This();
        pub const InputM=Matrix(f32,inputNodes,1);
        pub const WeightM=Matrix(f32,nodes, inputNodes);
        pub const BiasM=Matrix(f32,nodes,1);
        pub const InputC=inputNodes;
        pub const NodeC=nodes;
        weights:WeightM,
        biases:BiasM,
        pub fn init(iWeights:WeightM,iBiases:BiasM) Self{
            return Self {.weights = iWeights,.biases = iBiases};
        }
        pub fn calc(s:Self, input:InputM)BiasM{
            return s.weights.MatMul(input).add(s.biases);
        }
        pub fn backprop(s:*Self,input:InputM,dNext:BiasM,gamma:f32,epsilon:f32)InputM{
            var dLdW=dNext.MatMul(input.Transpose());
            const dLdX=s.weights.Transpose().MatMul(dNext);
            const delta=dLdW.ScalarMul(gamma);
            var newWeights=s.weights.sub(delta);
            var newBiases=s.biases.sub(dNext.ScalarMul(gamma));
            if(epsilon>0.97){
                newWeights=s.weights.add(delta);
                newBiases=s.biases.add(dNext.ScalarMul(gamma));
            }
            for (0..NodeC) |i| {
                for (0..InputC) |j| {
                    s.weights.data[i*InputC+j]=newWeights.data[i*InputC+j];
                }
                s.biases.data[i]=newBiases.data[i];
            }
            return dLdX;
        }
    };
}
pub fn ActivationLayer(
    comptime nodes: usize,
    comptime activation:(fn(x:f32)f32),
    comptime dActivation:(fn(x:f32)f32),
    ) type {
    return struct{
        const Self=@This();
        pub const LayerM=Matrix(f32,nodes,1);
        pub const Func=activation;
        pub const dFunc=dActivation;
        pub const NodeC=nodes;
        pub fn init() Self{
            return Self {};
        }
        pub fn calc(_:Self, input:LayerM)LayerM{
            return input.ApplyFunc(Func);
        }
        pub fn backprop(_:Self,dNext:LayerM)LayerM{
            return dNext.ApplyFunc(dFunc);
        }
        
    };
}
pub fn InputLayer(
    comptime nodes:usize
    ) type {
    return struct{
        const Self=@This();
        pub const LayerM=Matrix(f32,nodes,1);
        pub const NodeC=nodes;
        pub fn init() Self {
            return Self{};
        }
        pub fn calc(_:Self,inp:LayerM)LayerM{
            return inp;
        }
        pub fn backprop(_:Self,dNext:LayerM)LayerM{
            return dNext;
        }
    };
}
pub fn OutputLayer(
    comptime nodes:usize,
    comptime lossF:(fn(x:f32,y:f32)f32),
    comptime dLossF:(fn(x:f32,y:f32)f32),
) type {
    return struct{
        const Self=@This();
        pub const LayerM=Matrix(f32,nodes,1);
        pub const ErrFunc=lossF;
        pub const dErrFunc=dLossF;
        pub const NodeC=nodes;
        pub fn init() Self{
            return Self{};
        }
        pub fn calc(_:Self,input:LayerM)LayerM{
            return input;
        }
        pub fn calcLoss(_:Self,input:LayerM,expected:LayerM)f32{
            var out:f32=0.0;
            for (0..NodeC) |i| {
                out+=ErrFunc(input.data[i],expected.data[i]);
            }
            return out/@as(f32,@floatFromInt(NodeC));
        }
        pub fn backprop(_:Self,input:LayerM,expected:LayerM)LayerM{
            var out=LayerM.init().fill(0);
            for (0..NodeC) |i| {
                out.data[i]=dErrFunc(input.data[i],expected.data[i])/@as(f32,@floatFromInt(NodeC));
            }
            return out;
        }
    };
}
pub fn gammaX(x:f32)f32{
    return(std.math.exp(-x)*x*x);
}
pub fn gammaX2(x:f32)f32{
    return(std.math.pow(f32,(2.0*x+1.0)/(4.0*x),6.7331));
}
pub fn MeanSquareGen(A:type)(fn (B:A) A){
    return(struct{fn f(C: A) A{
        var g:A=C;
        g.a=(C.a-C.b)*(C.a-C.b);
        g.b=(C.a-C.b)*(C.a-C.b);
        return g;
    }}.f);
}
pub fn msd(x:f32,y:f32) f32 {
    return((x-y)*(x-y));
}
pub fn msd_dx(x:f32,y:f32) f32 {
    return(2.0*x-2.0*y);
}
pub fn sigmoid(x:f32) f32 {
    return(1/(1+std.math.exp(-x)));
}
pub fn sigmoid_dx(x:f32) f32 {
    return(std.math.exp(-x)/(2*std.math.exp(-x)+std.math.exp(-2*x)+1));
}
pub fn sigmoid_dx2(x:f32) f32 {
    return(1/(2+std.math.exp(-x)+std.math.exp(x)));
}
pub fn Testing(x:f32,y:f32,z:f32)Matrix(f32,3,1){
    comptime var Out=Matrix(f32,3,1);
    const out=Out.from_values([_]f32{
        z,
        x,
        y,
    });
    return out;
}
const LayerType=enum{
    Input,
    Activation,
    Hidden,
    Output,
};
pub fn loadingbar(curr:usize,max:f32,segments:f32)void{
    
    std.debug.print("\rProgress: |",.{});
    const a=@as(usize,@intFromFloat((@as(f32,@floatFromInt(curr))/max)*segments));
    for (0..a) |_| {
        std.debug.print("=",.{});
    }
    for (a..@as(usize,@intFromFloat(segments))) |_| {
        std.debug.print("-",.{});
    }
    std.debug.print("| {d}/{d} | {d:.3}%        \r",.{curr,max,@as(f32,@floatFromInt(curr))/max*100.0});
}
pub fn main() !void {
    // Prints to stderr (it's a shortcut based on `std.io.getStdErr()`)
    const Mat3x1=Matrix(f32,3,1);
    const Mat5x1=Matrix(f32,5,1);
    const Mat3x5=Matrix(f32,3,5);
    const Mat5x3=Matrix(f32,5,3);
    const Mat3x3=Matrix(f32,3,3);
    const InpL=InputLayer(3);
    const HL1=HiddenLayer(5, 3);
    const AL1=ActivationLayer(5, sigmoid, sigmoid_dx2);
    const HL2=HiddenLayer(3, 5);
    const AL2=ActivationLayer(3, sigmoid, sigmoid_dx);
    const HL3=HiddenLayer(3,3);
    const OutL=OutputLayer(3, msd, msd_dx);
    var il=InpL.init();
    var hl1=HL1.init(Mat5x3.from_values([_]f32{
        0.2,0.3,46.2,
        0.01,-0.24,0.9,
        1.0,0.43,0.0,
        1.2,2.34,5.0,
        1.32,1.37,1.36,
    }), Mat5x1.from_values([_]f32{
        0.2,
        0.43,
        0.09,
        0.234,
        0.4432,
    }));
    var al1=AL1.init();
    var hl2=HL2.init(Mat3x5.from_values([_]f32{
        1.434,1.4344,1.343,1.4434,1.543,
        1.090989876,1.57847,1.07832,1.45454,1.4345,
        1.75487,1.4773,0.33441,0.441,0.551,
    }), Mat3x1.from_values([_]f32{
        0.4351,
        0.545,
        0.43451,
    }));
    var al2=AL2.init();
    var hl3=HL3.init(Mat3x3.from_values([_]f32{
        0.32,0.20,0.52,
        0.54,0.01,0.11,
        0.55,0.36,0.91,
    }),Mat3x1.from_values([_]f32{
        0.03,
        0.04,
        0.09,
    }));
    var outl=OutL.init();
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    var gammaDiv:f32=1.0;
    const epochs:usize=1000000;
    var a1:Mat3x1=undefined;
    var a2:Mat5x1=undefined;
    var a3:Mat5x1=undefined;
    var a4:Mat3x1=undefined;
    var a5:Mat3x1=undefined;
    var a6:Mat3x1=undefined;
    var a7:Mat3x1=undefined;
    for (0..epochs) |i| {
        const epsilon=rand.float(f32);
        const x=rand.float(f32);
        const y=rand.float(f32);
        const z=rand.float(f32);
        a1=il.calc(Mat3x1.from_values([_]f32{x,y,z}));
        a2=hl1.calc(a1);
        a3=al1.calc(a2);
        a4=hl2.calc(a3);
        a5=al2.calc(a4);
        a6=hl3.calc(a5);
        a7=outl.calc(a6);
        const gamma=gammaX2(gammaDiv+1.0);
        const b0=outl.backprop(a7,Testing(x, y, z));
        const b1=(&hl3).backprop(a5, b0, gamma, epsilon);
        const b2=al2.backprop(b1);
        const b3=(&hl2).backprop(a3,b2,gamma,epsilon);
        const b4=al1.backprop(b3);
        _=(&hl1).backprop(a1,b4,gamma,epsilon);
        if(i==0){
            _=hl1.weights.print();
            var lossS:f32=0.0;
            for (0..1000) |_| {
                const x1=rand.float(f32);
                const y1=rand.float(f32);
                const z1=rand.float(f32);
                a1=il.calc(Mat3x1.from_values([_]f32{x1,y1,z1}));
                a2=hl1.calc(a1);
                a3=al1.calc(a2);
                a4=hl2.calc(a3);
                a5=al2.calc(a4);
                a6=hl3.calc(a5);
                a7=outl.calc(a6);
                lossS+=outl.calcLoss(a7,Testing(x1, y1, z1));
            }
            std.debug.print("\nLoss:{}\n",.{lossS/1000.0});
        }
        loadingbar(i+1, epochs, 100);
        if(i==epochs-1){
            _=hl1.weights.print();
            var lossS:f32=0.0;
            for (0..1000) |_| {
                const x1=rand.float(f32);
                const y1=rand.float(f32);
                const z1=rand.float(f32);
                a1=il.calc(Mat3x1.from_values([_]f32{x1,y1,z1}));
                a2=hl1.calc(a1);
                a3=al1.calc(a2);
                a4=hl2.calc(a3);
                a5=al2.calc(a4);
                a6=hl3.calc(a5);
                a7=outl.calc(a6);
                lossS+=outl.calcLoss(a7,Testing(x1, y1, z1));
            }
            std.debug.print("\nLoss:{}\n",.{lossS/1000.0});
        }
        gammaDiv+=1.0;

            //_=b5.print();
    } 
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();
    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try bw.flush(); // don't forget to flush!
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
