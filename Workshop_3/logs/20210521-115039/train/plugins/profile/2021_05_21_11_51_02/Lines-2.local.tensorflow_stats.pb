"?f
BHostIDLE"IDLE1    ???@A    ???@afϛA??ifϛA???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a/?$wL???i,??]%???Unknown?
iHostWriteSummary"WriteSummary(1      C@9      C@A      C@I      C@a?&?@?sl?iSr2$?????Unknown?
wHost_FusedMatMul"sequential_102/dense_316/Relu(1      ?@9      ?@A      ?@I      ?@a???6g?i???*ϲ???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ;@9      ;@A      :@I      :@a?"??wc?i????F????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      5@9      5@A      5@I      5@a
S_?|r_?i7?? ????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      1@9      1@A      1@I      1@a?@?uY?i?zԓ?????Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      (@9      (@A      (@I      (@a?
[QG?Q?i9(}??????Unknown
?	HostMatMul"-gradient_tape/sequential_102/dense_318/MatMul(1      &@9      &@A      &@I      &@a?t?
?xP?i?q?-?????Unknown
u
HostSigmoid" sequential_102/dense_318/Sigmoid(1      &@9      &@A      &@I      &@a?t?
?xP?i????/????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      1@9      1@A      $@I      $@aͼ??!?M?i???k????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@aͼ??!?M?i??K4)???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@aͼ??!?M?izm??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@aͼ??!?M?iiS?"???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?cyl??G?i?q*2  ???Unknown
dHostDataset"Iterator::Model(1      =@9      =@A       @I       @a?cyl??G?i?E?&???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?cyl??G?it?`,???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1       @9       @A       @I       @a?cyl??G?i??{y2???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a?cyl??G?i&???8???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?cyl??G?i	?S>???Unknown
?HostMatMul"-gradient_tape/sequential_102/dense_317/MatMul(1       @9       @A       @I       @a?cyl??G?i?'??D???Unknown
?HostMatMul"/gradient_tape/sequential_102/dense_317/MatMul_1(1       @9       @A       @I       @a?cyl??G?i1F?-J???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a\7????D?i? `?KO???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a\7????D?iM?׬?T???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a\7????D?i?uOl?Y???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a\7????D?ii0?+_???Unknown
?HostMatMul"-gradient_tape/sequential_102/dense_316/MatMul(1      @9      @A      @I      @a\7????D?i??>?Bd???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a\7????D?i?????i???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?
[QG?A?iH????m???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?
[QG?A?iS_?|r???Unknown
wHost_FusedMatMul"sequential_102/dense_317/Relu(1      @9      @A      @I      @a?
[QG?A?iΩ3??v???Unknown
z Host_FusedMatMul" sequential_102/dense_318/BiasAdd(1      @9      @A      @I      @a?
[QG?A?i? ?x{???Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aͼ??!?=?i??8V7???Unknown
v"HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aͼ??!?=?i??i??????Unknown
?#HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aͼ??!?=?iyٚ?????Unknown
?$HostBiasAddGrad":gradient_tape/sequential_102/dense_317/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aͼ??!?=?iq?˂r????Unknown
?%HostBiasAddGrad":gradient_tape/sequential_102/dense_318/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aͼ??!?=?ii???0????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?cyl??7?i?N??/????Unknown
|'HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a?cyl??7?i??T.????Unknown
?(HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?cyl??7?i?l?
-????Unknown
?)HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?cyl??7?i?2?+????Unknown
?*HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?cyl??7?iE??w*????Unknown
?+HostBiasAddGrad":gradient_tape/sequential_102/dense_316/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?cyl??7?iqN.)????Unknown
?,HostMatMul"/gradient_tape/sequential_102/dense_318/MatMul_1(1      @9      @A      @I      @a?cyl??7?i????'????Unknown
?-HostReadVariableOp"/sequential_102/dense_317/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?cyl??7?i?8i?&????Unknown
?.HostReadVariableOp".sequential_102/dense_317/MatMul/ReadVariableOp(1      @9      @A      @I      @a?cyl??7?i???Q%????Unknown
?/HostReadVariableOp".sequential_102/dense_318/MatMul/ReadVariableOp(1      @9      @A      @I      @a?cyl??7?i!W?$????Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?
[QG?1?i??nc????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a?
[QG?1?i??X?????Unknown
X2HostCast"Cast_3(1      @9      @A      @I      @a?
[QG?1?iD?B#?????Unknown
X3HostCast"Cast_5(1      @9      @A      @I      @a?
[QG?1?i?-, ????Unknown
\4HostGreater"Greater(1      @9      @A      @I      @a?
[QG?1?i05_????Unknown
?5HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      4@9      4@A      @I      @a?
[QG?1?ig[>?????Unknown
V6HostMean"Mean(1      @9      @A      @I      @a?
[QG?1?iȆ?Fݻ???Unknown
u7HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?
[QG?1?i)??O????Unknown
?8HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?
[QG?1?i?ݿX[????Unknown
z9HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?
[QG?1?i??a?????Unknown
~:HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?
[QG?1?iL4?j?????Unknown
v;HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?
[QG?1?i?_~s????Unknown
v<HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?
[QG?1?i?h|W????Unknown
?=HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?
[QG?1?io?R??????Unknown
?>HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?
[QG?1?i??<??????Unknown
??Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?
[QG?1?i1'?????Unknown
~@HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a?
[QG?1?i?8?S????Unknown
?AHostReadVariableOp"/sequential_102/dense_318/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?
[QG?1?i?c???????Unknown
VBHostCast"Cast(1       @9       @A       @I       @a?cyl??'?i?+B????Unknown
XCHostEqual"Equal(1       @9       @A       @I       @a?cyl??'?i??_?????Unknown
sDHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?cyl??'?i??Ϻ????Unknown
dEHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?cyl??'?iK??????Unknown
jFHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?cyl??'?i?I]q????Unknown
rGHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?cyl??'?iw?̎????Unknown
vHHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?cyl??'?i??'????Unknown
?IHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?cyl??'?i??1??????Unknown
}JHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?cyl??'?i9hx?????Unknown
`KHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?cyl??'?i?/?9?????Unknown
bLHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?cyl??'?ie??????Unknown
wMHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?cyl??'?i??L??????Unknown
?NHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?cyl??'?i???K
????Unknown
~OHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?cyl??'?i'Nڦ?????Unknown
?PHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?cyl??'?i?!	????Unknown
?QHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?cyl??'?iS?g]?????Unknown
?RHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?cyl??'?i餮?????Unknown
?SHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?cyl??'?il??????Unknown
?THostReluGrad"/gradient_tape/sequential_102/dense_316/ReluGrad(1       @9       @A       @I       @a?cyl??'?i4<o????Unknown
?UHostReluGrad"/gradient_tape/sequential_102/dense_317/ReluGrad(1       @9       @A       @I       @a?cyl??'?i???ʅ????Unknown
?VHostReadVariableOp"/sequential_102/dense_316/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?cyl??'?iA??%????Unknown
?WHostReadVariableOp".sequential_102/dense_316/MatMul/ReadVariableOp(1       @9       @A       @I       @a?cyl??'?i׊??????Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?cyl???i???.D????Unknown
XYHostCast"Cast_4(1      ??9      ??A      ??I      ??a?cyl???imRW?????Unknown
aZHostIdentity"Identity(1      ??9      ??A      ??I      ??a?cyl???i8????????Unknown?
?[HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a?cyl???i?7?????Unknown
T\HostMul"Mul(1      ??9      ??A      ??I      ??a?cyl???i?}A?B????Unknown
u]HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?cyl???i????????Unknown
y^HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?cyl???idE?@?????Unknown
x_HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a?cyl???i/?+??????Unknown
?`HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?cyl???i?ϛA????Unknown
?aHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?cyl???i?prI????Unknown
?bHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?cyl???i????????Unknown
?cHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?cyl???i[8???????Unknown
?dHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?cyl???i&?\R@????Unknown
?eHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?cyl???i?????????Unknown
JfHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown
WgHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i?????????Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
[iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown*?e
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@afffff&??ifffff&???Unknown?
iHostWriteSummary"WriteSummary(1      C@9      C@A      C@I      C@affffff??i????????Unknown?
wHost_FusedMatMul"sequential_102/dense_316/Relu(1      ?@9      ?@A      ?@I      ?@a?????̘?i?????????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ;@9      ;@A      :@I      :@a?????̔?ieffff????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      5@9      5@A      5@I      5@a?????̐?i????????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      1@9      1@A      1@I      1@a333333??i?????y???Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      (@9      (@A      (@I      (@a333333??ieffff????Unknown
?HostMatMul"-gradient_tape/sequential_102/dense_318/MatMul(1      &@9      &@A      &@I      &@a????????i????????Unknown
u	HostSigmoid" sequential_102/dense_318/Sigmoid(1      &@9      &@A      &@I      &@a????????i13333S???Unknown
?
HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      1@9      1@A      $@I      $@a      ??i13333????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a      ??i13333????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a      ??i13333???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a      ??i13333S???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a??????y?idffff????Unknown
dHostDataset"Iterator::Model(1      =@9      =@A       @I       @a??????y?i?????????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??????y?i?????????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1       @9       @A       @I       @a??????y?i????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a??????y?i03333S???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a??????y?icffff????Unknown
?HostMatMul"-gradient_tape/sequential_102/dense_317/MatMul(1       @9       @A       @I       @a??????y?i?????????Unknown
?HostMatMul"/gradient_tape/sequential_102/dense_317/MatMul_1(1       @9       @A       @I       @a??????y?i?????????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @affffffv?i????????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @affffffv?icffffF???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @affffffv?i03333s???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @affffffv?i?????????Unknown
?HostMatMul"-gradient_tape/sequential_102/dense_316/MatMul(1      @9      @A      @I      @affffffv?i?????????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @affffffv?i?????????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a333333s?i????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a333333s?icffffF???Unknown
wHost_FusedMatMul"sequential_102/dense_317/Relu(1      @9      @A      @I      @a333333s?i?????l???Unknown
zHost_FusedMatMul" sequential_102/dense_318/BiasAdd(1      @9      @A      @I      @a333333s?i/3333????Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a      p?i/3333????Unknown
v!HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a      p?i/3333????Unknown
?"HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a      p?i/3333????Unknown
?#HostBiasAddGrad":gradient_tape/sequential_102/dense_317/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a      p?i/3333???Unknown
?$HostBiasAddGrad":gradient_tape/sequential_102/dense_318/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a      p?i/33333???Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??????i?i?????L???Unknown
|&HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a??????i?icfffff???Unknown
?'HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??????i?i????????Unknown
?(HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??????i?i?????????Unknown
?)HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??????i?i13333????Unknown
?*HostBiasAddGrad":gradient_tape/sequential_102/dense_316/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??????i?i?????????Unknown
?+HostMatMul"/gradient_tape/sequential_102/dense_318/MatMul_1(1      @9      @A      @I      @a??????i?ieffff????Unknown
?,HostReadVariableOp"/sequential_102/dense_317/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??????i?i?????????Unknown
?-HostReadVariableOp".sequential_102/dense_317/MatMul/ReadVariableOp(1      @9      @A      @I      @a??????i?i????????Unknown
?.HostReadVariableOp".sequential_102/dense_318/MatMul/ReadVariableOp(1      @9      @A      @I      @a??????i?i333333???Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a333333c?ifffffF???Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a333333c?i?????Y???Unknown
X1HostCast"Cast_3(1      @9      @A      @I      @a333333c?i?????l???Unknown
X2HostCast"Cast_5(1      @9      @A      @I      @a333333c?i????????Unknown
\3HostGreater"Greater(1      @9      @A      @I      @a333333c?i23333????Unknown
?4HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      4@9      4@A      @I      @a333333c?ieffff????Unknown
V5HostMean"Mean(1      @9      @A      @I      @a333333c?i?????????Unknown
u6HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a333333c?i?????????Unknown
?7HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a333333c?i?????????Unknown
z8HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a333333c?i13333????Unknown
~9HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a333333c?idffff???Unknown
v:HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a333333c?i????????Unknown
v;HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a333333c?i?????,???Unknown
?<HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a333333c?i?????????Unknown
?=HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a333333c?i03333S???Unknown
?>Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a333333c?icfffff???Unknown
~?HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a333333c?i?????y???Unknown
?@HostReadVariableOp"/sequential_102/dense_318/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a333333c?i????̌???Unknown
VAHostCast"Cast(1       @9       @A       @I       @a??????Y?i?????????Unknown
XBHostEqual"Equal(1       @9       @A       @I       @a??????Y?icffff????Unknown
sCHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a??????Y?i03333????Unknown
dDHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??????Y?i?????????Unknown
jEHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a??????Y?i?????????Unknown
rFHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a??????Y?i?????????Unknown
vGHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??????Y?idffff????Unknown
?HHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??????Y?i13333????Unknown
}IHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a??????Y?i?????????Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??????Y?i????????Unknown
bKHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a??????Y?i????????Unknown
wLHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??????Y?ieffff&???Unknown
?MHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??????Y?i233333???Unknown
~NHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a??????Y?i?????????Unknown
?OHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a??????Y?i?????L???Unknown
?PHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??????Y?i?????Y???Unknown
?QHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a??????Y?iffffff???Unknown
?RHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a??????Y?i33333s???Unknown
?SHostReluGrad"/gradient_tape/sequential_102/dense_316/ReluGrad(1       @9       @A       @I       @a??????Y?i     ????Unknown
?THostReluGrad"/gradient_tape/sequential_102/dense_317/ReluGrad(1       @9       @A       @I       @a??????Y?i????̌???Unknown
?UHostReadVariableOp"/sequential_102/dense_316/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??????Y?i?????????Unknown
?VHostReadVariableOp".sequential_102/dense_316/MatMul/ReadVariableOp(1       @9       @A       @I       @a??????Y?igffff????Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??????I?i????̬???Unknown
XXHostCast"Cast_4(1      ??9      ??A      ??I      ??a??????I?i33333????Unknown
aYHostIdentity"Identity(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown?
?ZHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
T[HostMul"Mul(1      ??9      ??A      ??I      ??a??????I?ieffff????Unknown
u\HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
y]HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??????I?i13333????Unknown
x^HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
?_HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
?`HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??????I?icffff????Unknown
?aHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
?bHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??????I?i/3333????Unknown
?cHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
?dHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a??????I?i?????????Unknown
JeHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown
WfHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i?????????Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown2CPU