"?c
BHostIDLE"IDLE1     ??@A     ??@a?8=[???i?8=[????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      ]@9      ]@A      ]@I      ]@a?f{?a.??iUs$J?x???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      L@9      L@A      L@I      L@a??0?CG??io55Y?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?G@9     ?G@A     ?G@I     ?G@ax?_r=7??ių?N?2???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      <@9      <@A      8@I      8@a?_̪u?iн??^???Unknown
?HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      7@9      7@A      7@I      7@aZ%????t?i2?D?????Unknown
`HostGatherV2"
GatherV2_1(1      2@9      2@A      2@I      2@a1?CG@p?i??Vw%????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      2@9      2@A      2@I      2@a1?CG@p?i+A婥????Unknown
?	HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      1@9      1@A      1@I      1@a??????n?i?2??W????Unknown
o
HostSigmoid"sequential/dense_2/Sigmoid(1      1@9      1@A      1@I      1@a??????n?i?$??	???Unknown
^HostGatherV2"GatherV2(1      0@9      0@A      0@I      0@as\)??l?i?R?"???Unknown
vHostSub"%binary_crossentropy/logistic_loss/sub(1      0@9      0@A      0@I      0@as\)??l?i????????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      .@9      .@A      .@I      .@a?F?vk?i?f??Z???Unknown
dHostDataset"Iterator::Model(1     @`@9     @`@A      ,@I      ,@a??0?CGi?i??*?-t???Unknown
iHostWriteSummary"WriteSummary(1      ,@9      ,@A      ,@I      ,@a??0?CGi?i?u????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1      ,@9      ,@A      ,@I      ,@a??0?CGi?i?4?X?????Unknown
jHostMean"binary_crossentropy/Mean(1      *@9      *@A      *@I      *@aƚyg?ik??`5????Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      (@9      (@A      (@I      (@a?_̪e?iq?#-?????Unknown
XHostEqual"Equal(1      &@9      &@A      &@I      &@aEo???c?i?Cн?????Unknown
vHostExp"%binary_crossentropy/logistic_loss/Exp(1      &@9      &@A      &@I      &@aEo???c?i??|N?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a1?CG@`?i???g????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      "@9      "@A      "@I      "@a1?CG@`?i?:????Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a1?CG@`?iG~R?Y,???Unknown
VHostMean"Mean(1       @9       @A       @I       @as\)??\?iK,?w?:???Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @as\)??\?iO?{U=I???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @as\)??\?iS?3?W???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1       @9       @A       @I       @as\)??\?iW6?!f???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a??0?CGY?i?N???r???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??0?CGY?i?fiTh???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a??0?CGY?i K?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a??0?CGY?ic?-??????Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??0?CGY?i??:S????Unknown
a!HostCast"sequential/Cast(1      @9      @A      @I      @a??0?CGY?i?????????Unknown
v"HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?_̪U?ilJ!B̼???Unknown
?#HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?_̪U?i??P??????Unknown
y$HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a?_̪U?irO?w????Unknown
?%HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?_̪U?i?ѯtL????Unknown
?&HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a????TR?i??,?S????Unknown
?'HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a????TR?iy???Z????Unknown
V(HostSum"Sum_2(1      @9      @A      @I      @a????TR?i;?&?a????Unknown
v)HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a????TR?i???i???Unknown
}*HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a????TR?i?q Ip
???Unknown
?+HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????TR?i?^?sw???Unknown
t,Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a????TR?iCK?~???Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @as\)??L?iE?䌷#???Unknown
\.HostGreater"Greater(1      @9      @A      @I      @as\)??L?iG??{?*???Unknown
?/HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @as\)??L?iIPyj)2???Unknown
e0Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @as\)??L?iK?CYb9???Unknown?
s1HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @as\)??L?iM?H?@???Unknown
~2HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @as\)??L?iOU?6?G???Unknown
v3HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @as\)??L?iQ??%O???Unknown
b4HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @as\)??L?iSmFV???Unknown
?5Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @as\)??L?iUZ7]???Unknown
?6HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @as\)??L?iW???d???Unknown
?7HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @as\)??L?iY???k???Unknown
?8HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @as\)??L?i[_??)s???Unknown
?9HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @as\)??L?i]?`?bz???Unknown
?:HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @as\)??L?i_+??????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?_̪E?i??B`????Unknown
?<HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?_̪E?i??Zq????Unknown
z=HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?_̪E?i"Qr?ۑ???Unknown
~>HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?_̪E?ic?yF????Unknown
??HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?_̪E?i?ӡ,?????Unknown
?@HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?_̪E?i唹?????Unknown
?AHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a?_̪E?i&Vђ?????Unknown
tBHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @as\)??<?i??6
#????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @as\)??<?i(????????Unknown
VDHostCast"Cast(1       @9       @A       @I       @as\)??<?i?? ?[????Unknown
XEHostCast"Cast_2(1       @9       @A       @I       @as\)??<?i*fp?????Unknown
uFHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @as\)??<?i?/?甹???Unknown
dGHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @as\)??<?i,[0_1????Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @as\)??<?i?????????Unknown
xIHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @as\)??<?i.??Mj????Unknown
?JHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @as\)??<?i??_?????Unknown
?KHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @as\)??<?i0	?<?????Unknown
?LHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @as\)??<?i?4*??????Unknown
?MHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @as\)??<?i2`?+?????Unknown
}NHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @as\)??<?i????x????Unknown
OHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @as\)??<?i4?Y????Unknown
?PHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @as\)??<?i?⾑?????Unknown
XQHostCast"Cast_3(1      ??9      ??A      ??I      ??as\)??,?iuxq?????Unknown
XRHostCast"Cast_4(1      ??9      ??A      ??I      ??as\)??,?i5$	N????Unknown
|SHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??as\)??,?i???D????Unknown
rTHostAdd"!binary_crossentropy/logistic_loss(1      ??9      ??A      ??I      ??as\)??,?i?9???????Unknown
?UHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??as\)??,?iu?;??????Unknown
}VHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??as\)??,?i5e???????Unknown
`WHostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??as\)??,?i???3U????Unknown
uXHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??as\)??,?i??So#????Unknown
yYHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??as\)??,?iu&??????Unknown
?ZHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??as\)??,?i5????????Unknown
?[HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??as\)??,?i?Qk"?????Unknown
?\Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??as\)??,?i??^\????Unknown
?]HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??as\)??,?iu}Й*????Unknown
?^HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??as\)??,?i5???????Unknown
?_HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??as\)??,?i??5?????Unknown
?`HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??as\)??,?i?>?L?????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??as\)??,?iuԚ?c????Unknown
?bHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??as\)??,?i5jM?1????Unknown
~cHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??as\)??,?i?????????Unknown
IdHostAssignAddVariableOp"AssignAddVariableOp_1(i?????????Unknown
4eHostIdentity"Identity(i?????????Unknown?
'fHostMul"Mul(i?????????Unknown
JgHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown*?b
sHostDataset"Iterator::Model::ParallelMapV2(1      ]@9      ]@A      ]@I      ]@azq? ???izq? ????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      L@9      L@A      L@I      L@aS??????i?{???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?G@9     ?G@A     ?G@I     ?G@a??,?'???i?Iɢ}???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      <@9      <@A      8@I      8@a???t	6??iE???c????Unknown
?HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      7@9      7@A      7@I      7@a5v??<??i?2?ću???Unknown
`HostGatherV2"
GatherV2_1(1      2@9      2@A      2@I      2@a?FH/Q??i????????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      2@9      2@A      2@I      2@a?FH/Q??ip;x?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      1@9      1@A      1@I      1@au	68???i?{????Unknown
o	HostSigmoid"sequential/dense_2/Sigmoid(1      1@9      1@A      1@I      1@au	68???i??~??^???Unknown
^
HostGatherV2"GatherV2(1      0@9      0@A      0@I      0@a??#?a???ii9??f????Unknown
vHostSub"%binary_crossentropy/logistic_loss/sub(1      0@9      0@A      0@I      0@a??#?a???i4v??<???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      .@9      .@A      .@I      .@a??ҋC??i?G??:+???Unknown
dHostDataset"Iterator::Model(1     @`@9     @`@A      ,@I      ,@aS??????i2Bzq?????Unknown
iHostWriteSummary"WriteSummary(1      ,@9      ,@A      ,@I      ,@aS??????i?<?Y???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1      ,@9      ,@A      ,@I      ,@aS??????id7??#????Unknown
jHostMean"binary_crossentropy/Mean(1      *@9      *@A      *@I      *@aJ??ߏ??i?Iɢ}???Unknown
?HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      (@9      (@A      (@I      (@a???t	6??i?v?S????Unknown
XHostEqual"Equal(1      &@9      &@A      &@I      &@aj9??f???iȻ??4v???Unknown
vHostExp"%binary_crossentropy/logistic_loss/Exp(1      &@9      &@A      &@I      &@aj9??f???i? MJ????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a?FH/Q??i?!
?ZN???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      "@9      "@A      "@I      "@a?FH/Q??i?Bǻ?????Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a?FH/Q??i?c??????Unknown
VHostMean"Mean(1       @9       @A       @I       @a??#?a???i,?H|Xg???Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??#?a???i_?ν???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @a??#?a???i?ҋC???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1       @9       @A       @I       @a??#?a???iŠ??j???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aS??????i?b?_????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aS??????i]?.????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aS??????i?????M???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @aS??????i???nT????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aS??????iA??E?????Unknown
a HostCast"sequential/Cast(1      @9      @A      @I      @aS??????i??^?0???Unknown
v!HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a???t	6??i??1Bzq???Unknown
?"HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a???t	6??iYghR????Unknown
y#HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a???t	6??i??؍*????Unknown
?$HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???t	6??i%>??4???Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aտlm?{?i??(j???Unknown
?&HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aտlm?{?i%?a?????Unknown
V'HostSum"Sum_2(1      @9      @A      @I      @aտlm?{?i??<????Unknown
v(HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aտlm?{?i%??(???Unknown
})HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @aտlm?{?i?}??1B???Unknown
?*HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aտlm?{?i%W?p;x???Unknown
t+Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @aտlm?{?i?0??D????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??#?a?u?i>x??????Unknown
\-HostGreater"Greater(1      @9      @A      @I      @a??#?a?u?i׿lm????Unknown
?.HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a??#?a?u?ipO1?/???Unknown
e/Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a??#?a?u?i	O1?/[???Unknown?
s0HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a??#?a?u?i???j????Unknown
~1HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??#?a?u?i;??|?????Unknown
v2HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??#?a?u?i?%?@?????Unknown
b3HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??#?a?u?imm????Unknown
?4Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??#?a?u?i???U3???Unknown
?5HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??#?a?u?i??~??^???Unknown
?6HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??#?a?u?i8DaPˉ???Unknown
?7HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??#?a?u?iыC????Unknown
?8HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a??#?a?u?ij?%?@????Unknown
?9HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a??#?a?u?i?{???Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???t	6p?i?????+???Unknown
?;HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a???t	6p?ii???SL???Unknown
z<HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???t	6p?i<?Կl???Unknown
~=HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a???t	6p?i????+????Unknown
?>HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???t	6p?i?????????Unknown
??HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a???t	6p?i5]?????Unknown
?@HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a???t	6p?i?l p????Unknown
tAHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??#?a?e?i?6]????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??#?a?e?i?ZN?????Unknown
VCHostCast"Cast(1       @9       @A       @I       @a??#?a?e?iO~?FH/???Unknown
XDHostCast"Cast_2(1       @9       @A       @I       @a??#?a?e?i?0??D???Unknown
uEHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a??#?a?e?i??!
?Z???Unknown
dFHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??#?a?e?i??l p???Unknown
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??#?a?e?i?ν????Unknown
xHHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??#?a?e?iP1?/[????Unknown
?IHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a??#?a?e?iU???????Unknown
?JHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a??#?a?e?i?x???????Unknown
?KHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a??#?a?e?i???U3????Unknown
?LHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a??#?a?e?i?????????Unknown
}MHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a??#?a?e?iQ??n???Unknown
NHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a??#?a?e?i?{???Unknown
?OHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??#?a?e?i?+?ݨ2???Unknown
XPHostCast"Cast_3(1      ??9      ??A      ??I      ??a??#?a?U?iѽ??w=???Unknown
XQHostCast"Cast_4(1      ??9      ??A      ??I      ??a??#?a?U?i?O~?FH???Unknown
|RHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a??#?a?U?i??v?S???Unknown
rSHostAdd"!binary_crossentropy/logistic_loss(1      ??9      ??A      ??I      ??a??#?a?U?i?so??]???Unknown
?THostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a??#?a?U?iihR?h???Unknown
}UHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a??#?a?U?iO?`?s???Unknown
`VHostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a??#?a?U?i5)Y?O~???Unknown
uWHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??#?a?U?i?Qe????Unknown
yXHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??#?a?U?iMJ?????Unknown
?YHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a??#?a?U?i??Bǻ????Unknown
?ZHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??#?a?U?i?p;x?????Unknown
?[Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??#?a?U?i?4)Y????Unknown
?\HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a??#?a?U?i??,?'????Unknown
?]HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??#?a?U?i&%??????Unknown
?^HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??#?a?U?ie?<?????Unknown
?_HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??#?a?U?iKJ??????Unknown
?`HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??#?a?U?i1??b????Unknown
?aHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a??#?a?U?inO1????Unknown
~bHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a??#?a?U?i?????????Unknown
IcHostAssignAddVariableOp"AssignAddVariableOp_1(i?????????Unknown
4dHostIdentity"Identity(i?????????Unknown?
'eHostMul"Mul(i?????????Unknown
JfHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU