"?e
BHostIDLE"IDLE1     6?@A     6?@a?`??կ??i?`??կ???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@aT?~?????i"9???&???Unknown?
vHost_FusedMatMul"sequential_60/dense_186/Relu(1     ?E@9     ?E@A     ?E@I     ?E@a?a0?,\r?i????K???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(1      @@9      @@A      ?@I      ?@ax????xj?i?ٲ?f???Unknown
iHostWriteSummary"WriteSummary(1      <@9      <@A      <@I      <@a3??(?g?iE??
~???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      7@9      7@A      7@I      7@aj????c?iť?"?????Unknown
vHost_FusedMatMul"sequential_60/dense_187/Relu(1      5@9      5@A      5@I      5@a?????a?ir?? ?????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      4@9      4@A      4@I      4@a$C?8Aa?i?+?A?????Unknown
\	HostGreater"Greater(1      3@9      3@A      3@I      3@ac?\?9`?i?JF??????Unknown
s
HostDataset"Iterator::Model::ParallelMapV2(1      3@9      3@A      3@I      3@ac?\?9`?igi??????Unknown
?HostMatMul".gradient_tape/sequential_60/dense_187/MatMul_1(1      3@9      3@A      3@I      3@ac?\?9`?i@??.W????Unknown
VHostSum"Sum_2(1      ,@9      ,@A      ,@I      ,@a3??(?W?i	??K????Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      *@9      *@A      *@I      *@a??c?3V?ih>?e????Unknown
lHostIteratorGetNext"IteratorGetNext(1      &@9      &@A      &@I      &@a?g?z?R?i?Ŷw????Unknown
^HostGatherV2"GatherV2(1      "@9      "@A      "@I      "@aB?b??N?i??v?y???Unknown
?HostMatMul",gradient_tape/sequential_60/dense_187/MatMul(1      "@9      "@A      "@I      "@aB?b??N?icw6)???Unknown
?HostMatMul",gradient_tape/sequential_60/dense_188/MatMul(1      "@9      "@A      "@I      "@aB?b??N?iP?????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a:8???SK?ii????#???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a:8???SK?i?&?Ђ*???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a:8???SK?i???W1???Unknown
?HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a:8???SK?iS???,8???Unknown
dHostDataset"Iterator::Model(1      :@9      :@A      @I      @a3??(?G?i7???&>???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a3??(?G?i??2!D???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a3??(?G?i???|J???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a,?A??~D?iz??);O???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a,?A??~D?i???ZT???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a,?A??~D?ip??zY???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a,?A??~D?i?8C1?^???Unknown
?HostMatMul",gradient_tape/sequential_60/dense_186/MatMul(1      @9      @A      @I      @a,?A??~D?if?m޹c???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_60/dense_187/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a,?A??~D?i?Y???h???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a$C?8AA?i?|??m???Unknown
v HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a$C?8AA?i?4?cq???Unknown
v!HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a$C?8AA?iÂ??u???Unknown
?"HostBiasAddGrad"9gradient_tape/sequential_60/dense_188/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a$C?8AA?i%????y???Unknown
y#Host_FusedMatMul"sequential_60/dense_188/BiasAdd(1      @9      @A      @I      @a$C?8AA?i6	?2~???Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a:8???S;?iݾ?P?????Unknown
V%HostMean"Mean(1      @9      @A      @I      @a:8???S;?i?t?????Unknown
?&HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a:8???S;?i+*t7r????Unknown
~'HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a:8???S;?i????܋???Unknown
b(HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a:8???S;?iy?WG????Unknown
?)HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a:8???S;?i Kɑ?????Unknown
?*HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a:8???S;?i? ;????Unknown
?+HostReadVariableOp".sequential_60/dense_187/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a:8???S;?in??x?????Unknown
?,HostReadVariableOp"-sequential_60/dense_187/MatMul/ReadVariableOp(1      @9      @A      @I      @a:8???S;?il??????Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a,?A??~4?iR??????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a,?A??~4?i??H?????Unknown
X/HostCast"Cast_3(1      @9      @A      @I      @a,?A??~4?i?D?o?????Unknown
X0HostCast"Cast_5(1      @9      @A      @I      @a,?A??~4?i	?sF0????Unknown
X1HostEqual"Equal(1      @9      @A      @I      @a,?A??~4?iF??????Unknown
u2HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a,?A??~4?i???O????Unknown
?3HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a,?A??~4?i?e3?߮???Unknown
?4HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a,?A??~4?i??Ƞo????Unknown
z5HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a,?A??~4?i:?]w?????Unknown
u6HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a,?A??~4?iw>?M?????Unknown
?7HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a,?A??~4?i???$????Unknown
?8HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a,?A??~4?i????????Unknown
?9HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a,?A??~4?i.??>????Unknown
?:HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a,?A??~4?ik_H??????Unknown
?;HostBiasAddGrad"9gradient_tape/sequential_60/dense_186/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a,?A??~4?i???~^????Unknown
?<HostReadVariableOp".sequential_60/dense_186/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a,?A??~4?i??rU?????Unknown
?=HostReadVariableOp"-sequential_60/dense_186/MatMul/ReadVariableOp(1      @9      @A      @I      @a,?A??~4?i"8,~????Unknown
?>HostReadVariableOp".sequential_60/dense_188/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a,?A??~4?i_??????Unknown
t?HostSigmoid"sequential_60/dense_188/Sigmoid(1      @9      @A      @I      @a,?A??~4?i??2ٝ????Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a:8???S+?ip??S????Unknown
VAHostCast"Cast(1       @9       @A       @I       @a:8???S+?iD~?L????Unknown
sBHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a:8???S+?iY]??????Unknown
dCHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a:8???S+?i?3?r????Unknown
jDHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a:8???S+?i???'????Unknown
rEHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a:8???S+?i???3?????Unknown
vFHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a:8???S+?ih?@m?????Unknown
?GHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a:8???S+?i<???G????Unknown
`HHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a:8???S+?iz???????Unknown
wIHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a:8???S+?i?Tk?????Unknown
?JHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a:8???S+?i?/$Tg????Unknown
xKHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a:8???S+?i?
ݍ????Unknown
~LHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a:8???S+?i`????????Unknown
?MHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a:8???S+?i4?N?????Unknown
?NHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a:8???S+?i?;<????Unknown
?OHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1       @9       @A       @I       @a:8???S+?i?u?t?????Unknown
?PHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a:8???S+?i?Py??????Unknown
?QHostReluGrad".gradient_tape/sequential_60/dense_186/ReluGrad(1       @9       @A       @I       @a:8???S+?i?+2?[????Unknown
?RHostReluGrad".gradient_tape/sequential_60/dense_187/ReluGrad(1       @9       @A       @I       @a:8???S+?iX?!????Unknown
?SHostMatMul".gradient_tape/sequential_60/dense_188/MatMul_1(1       @9       @A       @I       @a:8???S+?i,??[?????Unknown
?THostReadVariableOp"-sequential_60/dense_188/MatMul/ReadVariableOp(1       @9       @A       @I       @a:8???S+?i ?\?{????Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a:8???S?ij)92V????Unknown
XVHostCast"Cast_4(1      ??9      ??A      ??I      ??a:8???S?iԖ?0????Unknown
aWHostIdentity"Identity(1      ??9      ??A      ??I      ??a:8???S?i>?k????Unknown?
?XHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??a:8???S?i?q??????Unknown
TYHostMul"Mul(1      ??9      ??A      ??I      ??a:8???S?iߪ??????Unknown
|ZHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a:8???S?i|L?B?????Unknown
v[HostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a:8???S?i??c?u????Unknown
}\HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a:8???S?iP'@|P????Unknown
w]HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a:8???S?i??+????Unknown
y^HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a:8???S?i$??????Unknown
?_HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a:8???S?i?o?R?????Unknown
?`Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a:8???S?i?ܱ??????Unknown
?aHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a:8???S?ibJ???????Unknown
?bHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a:8???S?i̷j)p????Unknown
?cHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a:8???S?i6%G?J????Unknown
?dHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a:8???S?i??#c%????Unknown
~eHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a:8???S?i     ???Unknown
ifHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i     ???Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown*?e
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@9     ?@A     ?@I     ?@a??{FJ??i??{FJ???Unknown?
vHost_FusedMatMul"sequential_60/dense_186/Relu(1     ?E@9     ?E@A     ?E@I     ?E@a???????i??6&{Z???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(1      @@9      @@A      ?@I      ?@ag???????i?~E?????Unknown
iHostWriteSummary"WriteSummary(1      <@9      <@A      <@I      <@aX?ܾ,??iE?)?H????Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      7@9      7@A      7@I      7@a?,Q~
7??i??0b???Unknown
vHost_FusedMatMul"sequential_60/dense_187/Relu(1      5@9      5@A      5@I      5@a?k%????i??G?????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      4@9      4@A      4@I      4@aKY?񢭏?i#?5?e???Unknown
\HostGreater"Greater(1      3@9      3@A      3@I      3@an??'??i??s?$????Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1      3@9      3@A      3@I      3@an??'??i??q?V???Unknown
?
HostMatMul".gradient_tape/sequential_60/dense_187/MatMul_1(1      3@9      3@A      3@I      3@an??'??iQ0<?????Unknown
VHostSum"Sum_2(1      ,@9      ,@A      ,@I      ,@aX?ܾ,??i?m??'???Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1      *@9      *@A      *@I      *@a>???C???if??y???Unknown
lHostIteratorGetNext"IteratorGetNext(1      &@9      &@A      &@I      &@a?W??Ll??ičM?????Unknown
^HostGatherV2"GatherV2(1      "@9      "@A      "@I      "@a?&@??|?i?ل??????Unknown
?HostMatMul",gradient_tape/sequential_60/dense_187/MatMul(1      "@9      "@A      "@I      "@a?&@??|?i?%??1???Unknown
?HostMatMul",gradient_tape/sequential_60/dense_188/MatMul(1      "@9      "@A      "@I      "@a?&@??|?i?q?V?j???Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a֭Z??Wy?i5'??f????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a֭Z??Wy?i?ܾ,????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a֭Z??Wy?i??ۗ????Unknown
?HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a֭Z??Wy?iIG?u5???Unknown
dHostDataset"Iterator::Model(1      :@9      :@A      @I      @aX?ܾ,v?i?e???a???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aX?ܾ,v?i??j?'????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aX?ܾ,v?iY?#|?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a`?*?s?i^+y?????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a`?*?s?ic?Μ????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a`?*?s?ih;$-?,???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a`?*?s?im?y??R???Unknown
?HostMatMul",gradient_tape/sequential_60/dense_186/MatMul(1      @9      @A      @I      @a`?*?s?irK?M?x???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_60/dense_187/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a`?*?s?iw?$ޖ????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aKY????o?i???D????Unknown
vHostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aKY????o?i)?$?????Unknown
v HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aKY????o?i???Ɵ????Unknown
?!HostBiasAddGrad"9gradient_tape/sequential_60/dense_188/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aKY????o?iۘ?iM???Unknown
y"Host_FusedMatMul"sequential_60/dense_188/BiasAdd(1      @9      @A      @I      @aKY????o?i4???<???Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a֭Z??Wi?i??l?RV???Unknown
V$HostMean"Mean(1      @9      @A      @I      @a֭Z??Wi?i???w?o???Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a֭Z??Wi?i>??-????Unknown
~&HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a֭Z??Wi?i???Y????Unknown
b'HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a֭Z??Wi?i?O???????Unknown
?(HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a֭Z??Wi?iH?4N	????Unknown
?)HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a֭Z??Wi?i??a????Unknown
?*HostReadVariableOp".sequential_60/dense_187/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a֭Z??Wi?i?_Q?????Unknown
?+HostReadVariableOp"-sequential_60/dense_187/MatMul/ReadVariableOp(1      @9      @A      @I      @a֭Z??Wi?iR??n!???Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a`?*?c?iT~
74???Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a`?*?c?iVB5?G???Unknown
X.HostCast"Cast_3(1      @9      @A      @I      @a`?*?c?iX`?Z???Unknown
X/HostCast"Cast_5(1      @9      @A      @I      @a`?*?c?iZʊ?m???Unknown
X0HostEqual"Equal(1      @9      @A      @I      @a`?*?c?i\??W????Unknown
u1HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a`?*?c?i^R?????Unknown
?2HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a`?*?c?i`?????Unknown
?3HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a`?*?c?ib?5?????Unknown
z4HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a`?*?c?id?`x ????Unknown
u5HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a`?*?c?ifb?@"????Unknown
?6HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a`?*?c?ih&?$????Unknown
?7HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a`?*?c?ij???%???Unknown
?8HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a`?*?c?il??'???Unknown
?9HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a`?*?c?inr6a)+???Unknown
?:HostBiasAddGrad"9gradient_tape/sequential_60/dense_186/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a`?*?c?ip6a)+>???Unknown
?;HostReadVariableOp".sequential_60/dense_186/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a`?*?c?ir???,Q???Unknown
?<HostReadVariableOp"-sequential_60/dense_186/MatMul/ReadVariableOp(1      @9      @A      @I      @a`?*?c?it???.d???Unknown
?=HostReadVariableOp".sequential_60/dense_188/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a`?*?c?iv???0w???Unknown
t>HostSigmoid"sequential_60/dense_188/Sigmoid(1      @9      @A      @I      @a`?*?c?ixFJ2????Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a֭Z??WY?i?s?$ޖ???Unknown
V@HostCast"Cast(1       @9       @A       @I       @a֭Z??WY?i&????????Unknown
sAHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a֭Z??WY?i}?a?5????Unknown
dBHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a֭Z??WY?i??(??????Unknown
jCHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a֭Z??WY?i+)???????Unknown
rDHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a֭Z??WY?i?V?j9????Unknown
vEHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a֭Z??WY?iك~E?????Unknown
?FHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a֭Z??WY?i0?E ?????Unknown
`GHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a֭Z??WY?i???<????Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a֭Z??WY?i???????Unknown
?IHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a֭Z??WY?i59??????Unknown
xJHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a֭Z??WY?i?fb?@"???Unknown
~KHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a֭Z??WY?i??)f?.???Unknown
?LHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a֭Z??WY?i:??@?;???Unknown
?MHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a֭Z??WY?i???DH???Unknown
?NHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1       @9       @A       @I       @a֭Z??WY?i???T???Unknown
?OHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a֭Z??WY?i?IFћa???Unknown
?PHostReluGrad".gradient_tape/sequential_60/dense_186/ReluGrad(1       @9       @A       @I       @a֭Z??WY?i?v?Gn???Unknown
?QHostReluGrad".gradient_tape/sequential_60/dense_187/ReluGrad(1       @9       @A       @I       @a֭Z??WY?i??Ԇ?z???Unknown
?RHostMatMul".gradient_tape/sequential_60/dense_188/MatMul_1(1       @9       @A       @I       @a֭Z??WY?iDћa?????Unknown
?SHostReadVariableOp"-sequential_60/dense_188/MatMul/ReadVariableOp(1       @9       @A       @I       @a֭Z??WY?i??b<K????Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a֭Z??WI?iF??)?????Unknown
XUHostCast"Cast_4(1      ??9      ??A      ??I      ??a֭Z??WI?i?+*?????Unknown
aVHostIdentity"Identity(1      ??9      ??A      ??I      ??a֭Z??WI?i?M????Unknown?
?WHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??a֭Z??WI?iGY???????Unknown
TXHostMul"Mul(1      ??9      ??A      ??I      ??a֭Z??WI?i??T??????Unknown
|YHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a֭Z??WI?i????N????Unknown
vZHostExp"%binary_crossentropy/logistic_loss/Exp(1      ??9      ??A      ??I      ??a֭Z??WI?iH??????Unknown
}[HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a֭Z??WI?i????????Unknown
w\HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a֭Z??WI?i?J??P????Unknown
y]HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a֭Z??WI?iI?F??????Unknown
?^HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a֭Z??WI?i?w?o?????Unknown
?_Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a֭Z??WI?i?]R????Unknown
?`HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a֭Z??WI?iJ?qJ?????Unknown
?aHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a֭Z??WI?i?;?7?????Unknown
?bHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a֭Z??WI?i??8%T????Unknown
?cHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a֭Z??WI?iKi??????Unknown
~dHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a֭Z??WI?i?????????Unknown
ieHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
[fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown2CPU