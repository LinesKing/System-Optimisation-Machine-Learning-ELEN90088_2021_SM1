"?d
BHostIDLE"IDLE1    ???@A    ???@a????Z??i????Z???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     x?@9     x?@A     x?@I     x?@a??????i?.???{???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      C@9      C@A      C@I      C@aK?????k?i?%Z?h????Unknown
sHostSigmoid"sequential_13/dense_43/Sigmoid(1      B@9      B@A      B@I      B@ao????j?iI??l?????Unknown
iHostWriteSummary"WriteSummary(1      ;@9      ;@A      ;@I      ;@a?"?h??c?il?NP????Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      =@9      =@A      9@I      9@a???Dj#b?i$T??A????Unknown
uHost_FusedMatMul"sequential_13/dense_41/Relu(1      3@9      3@A      3@I      3@aK?????[?i?Ok?
????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      2@I      2@ao????Z?i]0??????Unknown
|	HostSelect"(binary_crossentropy/logistic_loss/Select(1      1@9      1@A      1@I      1@a??g?X?ii?d?o????Unknown
?
HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      .@9      .@A      .@I      .@a?B!?U?i
???Q	???Unknown
VHostCast"Cast(1      (@9      (@A      (@I      (@aJ???iQ?i???d???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a&Y,?vM?i?҂?G???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@ao????J?iåc????Unknown
?HostMatMul"-gradient_tape/sequential_13/dense_42/MatMul_1(1      "@9      "@A      "@I      "@ao????J?ic??W&???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@ao????J?iģ???,???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a??VC?7G?ioy|??2???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??VC?7G?iOoz8???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1       @9       @A       @I       @a??VC?7G?i?$?SH>???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a??VC?7G?ip?.8D???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a??VC?7G?iп?I???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a????PD?i??D?N???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a????PD?iF?lT???Unknown
dHostDataset"Iterator::Model(1     ?F@9     ?F@A      @I      @a????PD?i? ?? Y???Unknown
~HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a????PD?i󻺼4^???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a????PD?i?v??Hc???Unknown
HostMatMul"+gradient_tape/sequential_13/dense_41/MatMul(1      @9      @A      @I      @a????PD?i?1?]h???Unknown
HostMatMul"+gradient_tape/sequential_13/dense_42/MatMul(1      @9      @A      @I      @a????PD?i???4qm???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aJ???iA?i?#??q???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aJ???iA?iW-?&v???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aJ???iA?i???v?z???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_13/dense_43/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aJ???iA?i?mi??~???Unknown
 HostMatMul"+gradient_tape/sequential_13/dense_43/MatMul(1      @9      @A      @I      @aJ???iA?i?M5????Unknown
\!HostGreater"Greater(1      @9      @A      @I      @a&Y,?v=?i????Ն???Unknown
V"HostSum"Sum_2(1      @9      @A      @I      @a&Y,?v=?i0??v????Unknown
v#HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a&Y,?v=?i??eZ????Unknown
?$HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a&Y,?v=?iF$@	?????Unknown
?%HostBiasAddGrad"8gradient_tape/sequential_13/dense_41/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a&Y,?v=?iѩ?X????Unknown
?&HostBiasAddGrad"8gradient_tape/sequential_13/dense_42/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a&Y,?v=?i\/?f?????Unknown
u'Host_FusedMatMul"sequential_13/dense_42/Relu(1      @9      @A      @I      @a&Y,?v=?i????????Unknown
x(Host_FusedMatMul"sequential_13/dense_43/BiasAdd(1      @9      @A      @I      @a&Y,?v=?ir:??:????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??VC?77?iH???!????Unknown
X*HostEqual"Equal(1      @9      @A      @I      @a??VC?77?i;?????Unknown
?+HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a??VC?77?i?z???????Unknown
V,HostMean"Mean(1      @9      @A      @I      @a??VC?77?i??ˍ֫???Unknown
s-HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a??VC?77?i?P??????Unknown
?.HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??VC?77?iv?\r?????Unknown
v/HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??VC?77?iL&?d?????Unknown
v0HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??VC?77?i"??Vr????Unknown
?1HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a??VC?77?i??5IY????Unknown
?2Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??VC?77?i?f~;@????Unknown
?3HostMatMul"-gradient_tape/sequential_13/dense_43/MatMul_1(1      @9      @A      @I      @a??VC?77?i???-'????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aJ???i1?i?!}cT????Unknown
X5HostCast"Cast_3(1      @9      @A      @I      @aJ???i1?i?q3??????Unknown
?6HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aJ???i1?i??ή????Unknown
z7HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @aJ???i1?i$??????Unknown
?8HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @aJ???i1?iDbV:	????Unknown
b9HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aJ???i1?id?p6????Unknown
?:HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @aJ???i1?i?åc????Unknown
?;HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @aJ???i1?i?Ryې????Unknown
?<HostReadVariableOp"-sequential_13/dense_42/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aJ???i1?iĢ/?????Unknown
?=HostReadVariableOp"-sequential_13/dense_43/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aJ???i1?i???F?????Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??VC?7'?iO(
?^????Unknown
X?HostCast"Cast_4(1       @9       @A       @I       @a??VC?7'?i?].9?????Unknown
u@HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a??VC?7'?i%?R?E????Unknown
|AHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??VC?7'?i??v+?????Unknown
dBHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??VC?7'?i????,????Unknown
rCHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a??VC?7'?if3??????Unknown
vDHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??VC?7'?i?h??????Unknown
~EHostSelect"*binary_crossentropy/logistic_loss/Select_1(1       @9       @A       @I       @a??VC?7'?i<??????Unknown
vFHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a??VC?7'?i??+??????Unknown
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??VC?7'?i	Pn????Unknown
?HHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??VC?7'?i}>t{?????Unknown
xIHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??VC?7'?i?s??T????Unknown
?JHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a??VC?7'?iS??m?????Unknown
?KHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a??VC?7'?i????;????Unknown
?LHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??VC?7'?i)`?????Unknown
?MHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a??VC?7'?i?I)?"????Unknown
?NHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a??VC?7'?i?~MR?????Unknown
?OHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a??VC?7'?ij?q?	????Unknown
?PHostReadVariableOp",sequential_13/dense_41/MatMul/ReadVariableOp(1       @9       @A       @I       @a??VC?7'?i???D}????Unknown
?QHostReadVariableOp",sequential_13/dense_42/MatMul/ReadVariableOp(1       @9       @A       @I       @a??VC?7'?i@???????Unknown
tRHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a??VC?7?i?9Lz?????Unknown
vSHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??VC?7?i?T?6d????Unknown
XTHostCast"Cast_5(1      ??9      ??A      ??I      ??a??VC?7?i_op?????Unknown
aUHostIdentity"Identity(1      ??9      ??A      ??I      ??a??VC?7?i???????Unknown?
jVHostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??a??VC?7?iɤ?l?????Unknown
}WHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a??VC?7?i~?&)K????Unknown
`XHostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a??VC?7?i3ڸ?????Unknown
uYHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??VC?7?i??J??????Unknown
wZHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??VC?7?i??^x????Unknown
y[HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??VC?7?iR*o2????Unknown
?\HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??VC?7?iE??????Unknown
?]Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??VC?7?i?_???????Unknown
?^HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??VC?7?iqz%Q_????Unknown
?_HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??VC?7?i&??????Unknown
~`HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a??VC?7?iۯI??????Unknown
?aHostReluGrad"-gradient_tape/sequential_13/dense_41/ReluGrad(1      ??9      ??A      ??I      ??a??VC?7?i??ۆ?????Unknown
?bHostReluGrad"-gradient_tape/sequential_13/dense_42/ReluGrad(1      ??9      ??A      ??I      ??a??VC?7?iE?mCF????Unknown
?cHostReadVariableOp"-sequential_13/dense_41/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??VC?7?i?????????Unknown
?dHostReadVariableOp",sequential_13/dense_43/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??VC?7?iXI?\ ???Unknown
'eHostMul"Mul(iXI?\ ???Unknown
WfHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(iXI?\ ???Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(iXI?\ ???Unknown
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(iXI?\ ???Unknown*?d
uHostFlushSummaryWriter"FlushSummaryWriter(1     x?@9     x?@A     x?@I     x?@a~?	B???i~?	B????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      C@9      C@A      C@I      C@a6??L<??i(QX?"????Unknown
sHostSigmoid"sequential_13/dense_43/Sigmoid(1      B@9      B@A      B@I      B@a??^????i5H?4H????Unknown
iHostWriteSummary"WriteSummary(1      ;@9      ;@A      ;@I      ;@a-)D?{??i~??f$/???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      =@9      =@A      9@I      9@a?U??&???i*t??E????Unknown
uHost_FusedMatMul"sequential_13/dense_41/Relu(1      3@9      3@A      3@I      3@a6??L<??i????5G???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      2@I      2@a??^????i?ѳȹ???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      1@9      1@A      1@I      1@a???S??i????%???Unknown
?	HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      .@9      .@A      .@I      .@a?f$/?އ?iXx?Wx????Unknown
V
HostCast"Cast(1      (@9      (@A      (@I      (@a???Xw??i ?4?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@aڈ0?q??i?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a??^??|?i?>~?J???Unknown
?HostMatMul"-gradient_tape/sequential_13/dense_42/MatMul_1(1      "@9      "@A      "@I      "@a??^??|?i??&?????Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a??^??|?ib?1J]????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @aH:?v?uy?i??3I????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @aH:?v?uy?iL?5#???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1       @9       @A       @I       @aH:?v?uy?i?	?!V???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @aH:?v?uy?i6$??????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @aH:?v?uy?i?>???????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a????5Gv?iѵ?B?????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a????5Gv?i?,r????Unknown
dHostDataset"Iterator::Model(1     ?F@9     ?F@A      @I      @a????5Gv?i?A?A???Unknown
~HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a????5Gv?iC?2n???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a????5Gv?ii????????Unknown
HostMatMul"+gradient_tape/sequential_13/dense_41/MatMul(1      @9      @A      @I      @a????5Gv?i?	?]O????Unknown
HostMatMul"+gradient_tape/sequential_13/dense_42/MatMul(1      @9      @A      @I      @a????5Gv?i????????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a???Xws?i?T1????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a???Xws?ic(???@???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a???Xws?i:???pf???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_13/dense_43/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???Xws?i?F??????Unknown
HostMatMul"+gradient_tape/sequential_13/dense_43/MatMul(1      @9      @A      @I      @a???Xws?i???rҲ???Unknown
\ HostGreater"Greater(1      @9      @A      @I      @aڈ0?q?o?iqԌ??????Unknown
V!HostSum"Sum_2(1      @9      @A      @I      @aڈ0?q?o?i?!Vy????Unknown
v"HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aڈ0?q?o?i?5??L???Unknown
?#HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aڈ0?q?o?ifI9 2???Unknown
?$HostBiasAddGrad"8gradient_tape/sequential_13/dense_41/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aڈ0?q?o?i??ݪ?Q???Unknown
?%HostBiasAddGrad"8gradient_tape/sequential_13/dense_42/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aڈ0?q?o?i?q?q???Unknown
u&Host_FusedMatMul"sequential_13/dense_42/Relu(1      @9      @A      @I      @aڈ0?q?o?i????????Unknown
x'Host_FusedMatMul"sequential_13/dense_43/BiasAdd(1      @9      @A      @I      @aڈ0?q?o?i0(??m????Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aH:?v?ui?ij???????Unknown
X)HostEqual"Equal(1      @9      @A      @I      @aH:?v?ui?i?B??Y????Unknown
?*HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @aH:?v?ui?i?????????Unknown
V+HostMean"Mean(1      @9      @A      @I      @aH:?v?ui?i]t?E???Unknown
s,HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aH:?v?ui?iR??Ż0???Unknown
?-HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @aH:?v?ui?i?wa?1J???Unknown
v.HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aH:?v?ui?i?خ?c???Unknown
v/HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aH:?v?ui?i ?N?}???Unknown
?0HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aH:?v?ui?i:ŗ?????Unknown
?1Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @aH:?v?ui?it?;?	????Unknown
?2HostMatMul"-gradient_tape/sequential_13/dense_43/MatMul_1(1      @9      @A      @I      @aH:?v?ui?i?9??????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???Xwc?i?#??????Unknown
X4HostCast"Cast_3(1      @9      @A      @I      @a???Xwc?i?do?????Unknown
?5HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a???Xwc?ir???????Unknown
z6HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???Xwc?i^?^????Unknown
?7HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a???Xwc?iJ?n??(???Unknown
b8HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a???Xwc?i6??L<???Unknown
?9HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a???Xwc?i"? ?*O???Unknown
?:HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a???Xwc?i?y;Cb???Unknown
?;HostReadVariableOp"-sequential_13/dense_42/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???Xwc?i?rҲ[u???Unknown
?<HostReadVariableOp"-sequential_13/dense_43/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???Xwc?i?\+*t????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aH:?v?uY?i??f$/????Unknown
X>HostCast"Cast_4(1       @9       @A       @I       @aH:?v?uY?i ???????Unknown
u?HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @aH:?v?uY?i?0??????Unknown
|@HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @aH:?v?uY?iZw`????Unknown
dAHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @aH:?v?uY?i??S????Unknown
rBHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aH:?v?uY?i???????Unknown
vCHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aH:?v?uY?i1K??????Unknown
~DHostSelect"*binary_crossentropy/logistic_loss/Select_1(1       @9       @A       @I       @aH:?v?uY?iΑ?K????Unknown
vEHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aH:?v?uY?ik?@?????Unknown
wFHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aH:?v?uY?i|?????Unknown
?GHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @aH:?v?uY?i?e??|???Unknown
xHHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @aH:?v?uY?iB???7!???Unknown
?IHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @aH:?v?uY?i??-??-???Unknown
?JHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @aH:?v?uY?i|9i٭:???Unknown
?KHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aH:?v?uY?i???hG???Unknown
?LHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @aH:?v?uY?i????#T???Unknown
?MHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @aH:?v?uY?iS??`???Unknown
?NHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aH:?v?uY?i?SVm???Unknown
?OHostReadVariableOp",sequential_13/dense_41/MatMul/ReadVariableOp(1       @9       @A       @I       @aH:?v?uY?i????Tz???Unknown
?PHostReadVariableOp",sequential_13/dense_42/MatMul/ReadVariableOp(1       @9       @A       @I       @aH:?v?uY?i*?̶????Unknown
tQHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??aH:?v?uI?iy??3m????Unknown
vRHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aH:?v?uI?i?'?ʓ???Unknown
XSHostCast"Cast_5(1      ??9      ??A      ??I      ??aH:?v?uI?i?%.(????Unknown
aTHostIdentity"Identity(1      ??9      ??A      ??I      ??aH:?v?uI?ifnC??????Unknown?
jUHostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??aH:?v?uI?i?a(?????Unknown
}VHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??aH:?v?uI?i?~?@????Unknown
`WHostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??aH:?v?uI?iSX?"?????Unknown
uXHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??aH:?v?uI?i?????????Unknown
wYHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aH:?v?uI?i???Y????Unknown
yZHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aH:?v?uI?i@B???????Unknown
?[HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aH:?v?uI?i??????Unknown
?\Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aH:?v?uI?iވ0?q????Unknown
?]HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aH:?v?uI?i-,N?????Unknown
?^HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??aH:?v?uI?i|?k?,????Unknown
~_HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??aH:?v?uI?i?r??????Unknown
?`HostReluGrad"-gradient_tape/sequential_13/dense_41/ReluGrad(1      ??9      ??A      ??I      ??aH:?v?uI?i???????Unknown
?aHostReluGrad"-gradient_tape/sequential_13/dense_42/ReluGrad(1      ??9      ??A      ??I      ??aH:?v?uI?ii??E????Unknown
?bHostReadVariableOp"-sequential_13/dense_41/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aH:?v?uI?i?\₢????Unknown
?cHostReadVariableOp",sequential_13/dense_43/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aH:?v?uI?i     ???Unknown
'dHostMul"Mul(i     ???Unknown
WeHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i     ???Unknown
[fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown2CPU