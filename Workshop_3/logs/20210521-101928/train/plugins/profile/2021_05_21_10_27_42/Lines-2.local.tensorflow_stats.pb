"?P
BHostIDLE"IDLE1     #?@A     #?@a?O?????i?O??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     (?@9     (?@A     (?@I     (?@aQ??Z@??i?RpX?)???Unknown?
dHostDataset"Iterator::Model(1     ?F@9     ?F@A      9@I      9@a???$?mj?i?(?9?C???Unknown
iHostWriteSummary"WriteSummary(1      9@9      9@A      9@I      9@a???$?mj?i???k^???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      8@9      8@A      7@I      7@aUY???Ph?i:????v???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@a??w?$e?i/'?j?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      3@9      3@A      3@I      3@ag(e?d?iW?H|?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      3@9      3@A      3@I      3@ag(e?d?i?ʍ????Unknown
u	Host_FusedMatMul"sequential_25/dense_78/Relu(1      2@9      2@A      2@I      2@a,\R?nc?i?C??????Unknown
\
HostGreater"Greater(1      *@9      *@A      *@I      *@a?迃|[?i:8?>?????Unknown
HostMatMul"+gradient_tape/sequential_25/dense_79/MatMul(1      $@9      $@A      $@I      $@a??w?$U?i4???d????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a,\R?nS?ib?P?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a,\R?nS?i?F?l????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      "@9      "@A      "@I      "@a,\R?nS?i?????????Unknown
?HostMatMul"-gradient_tape/sequential_25/dense_79/MatMul_1(1      "@9      "@A      "@I      "@a,\R?nS?i??ovs???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a,\R?nS?iB?-????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??,?)?P?i|ػBl???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a??,?)?P?i?n?W????Unknown
HostMatMul"+gradient_tape/sequential_25/dense_80/MatMul(1       @9       @A       @I       @a??,?)?P?i@mlV(???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a}V?șM?iֈ?޼/???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a}V?șM?il?P#7???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_25/dense_80/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a}V?șM?i?%É>???Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a}V?șM?i?c5?E???Unknown
xHost_FusedMatMul"sequential_25/dense_80/BiasAdd(1      @9      @A      @I      @a}V?șM?i.???VM???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?%É>_I?i?Cw?S???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?%É>_I?i?x?FZ???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?%É>_I?i???^`???Unknown
HostMatMul"+gradient_tape/sequential_25/dense_78/MatMul(1      @9      @A      @I      @a?%É>_I?iRZ*??f???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??w?$E?iO?1?k???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??w?$E?iL9@Hq???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??w?$E?iIt@m?v???Unknown
? HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a??w?$E?iF?G??{???Unknown
?!HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a??w?$E?iC0O?#????Unknown
?"HostBiasAddGrad"8gradient_tape/sequential_25/dense_78/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??w?$E?i@?V?l????Unknown
?#HostBiasAddGrad"8gradient_tape/sequential_25/dense_79/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??w?$E?i=?]!?????Unknown
u$HostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a??w?$E?i:JeN?????Unknown
u%Host_FusedMatMul"sequential_25/dense_79/Relu(1      @9      @A      @I      @a??w?$E?i7?l{H????Unknown
u&HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a??,?)?@?ih???????Unknown
i'HostMean"mean_squared_error/Mean(1      @9      @A      @I      @a??,?)?@?i?>E??????Unknown
?(HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a??,?)?@?iʉ??????Unknown
?)HostReadVariableOp"-sequential_25/dense_78/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??,?)?@?i???2????Unknown
?*HostReadVariableOp"-sequential_25/dense_80/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??,?)?@?i, ?/m????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?%É>_9?i?X[?????Unknown
V,HostCast"Cast(1      @9      @A      @I      @a?%É>_9?i??,?ı???Unknown
X-HostCast"Cast_3(1      @9      @A      @I      @a?%É>_9?i[????????Unknown
`.HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?%É>_9?i???????Unknown
b/HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?%É>_9?i%:??H????Unknown
}0HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @a?%É>_9?i?rq?t????Unknown
?1HostReluGrad"-gradient_tape/sequential_25/dense_79/ReluGrad(1      @9      @A      @I      @a?%É>_9?i??B??????Unknown
?2HostMatMul"-gradient_tape/sequential_25/dense_80/MatMul_1(1      @9      @A      @I      @a?%É>_9?iT?n?????Unknown
?3HostReadVariableOp"-sequential_25/dense_79/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?%É>_9?i??U?????Unknown
s4HostSigmoid"sequential_25/dense_80/Sigmoid(1      @9      @A      @I      @a?%É>_9?iT?=$????Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??,?)?0?i?y??A????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a??,?)?0?iN?"?^????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??,?)?0?i??X|????Unknown
X8HostEqual"Equal(1       @9       @A       @I       @a??,?)?0?i~??R?????Unknown
V9HostMean"Mean(1       @9       @A       @I       @a??,?)?0?iŗ?????Unknown
T:HostMul"Mul(1       @9       @A       @I       @a??,?)?0?i?5???????Unknown
s;HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a??,?)?0?iF[1"?????Unknown
|<HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??,?)?0?iހgg????Unknown
w=HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??,?)?0?iv???+????Unknown
?>HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a??,?)?0?i???H????Unknown
w?HostCast"%gradient_tape/mean_squared_error/Cast(1       @9       @A       @I       @a??,?)?0?i??	7f????Unknown
u@HostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @a??,?)?0?i>@|?????Unknown
AHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1       @9       @A       @I       @a??,?)?0?i?<v??????Unknown
uBHostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a??,?)?0?inb??????Unknown
}CHostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a??,?)?0?i??K?????Unknown
?DHostReluGrad"-gradient_tape/sequential_25/dense_78/ReluGrad(1       @9       @A       @I       @a??,?)?0?i????????Unknown
?EHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??,?)?0?i6?N?????Unknown
|FHostDivNoNan"&mean_squared_error/weighted_loss/value(1       @9       @A       @I       @a??,?)?0?i???3????Unknown
?GHostReadVariableOp",sequential_25/dense_78/MatMul/ReadVariableOp(1       @9       @A       @I       @a??,?)?0?if?`P????Unknown
?HHostReadVariableOp",sequential_25/dense_79/MatMul/ReadVariableOp(1       @9       @A       @I       @a??,?)?0?i?C??m????Unknown
XIHostCast"Cast_4(1      ??9      ??A      ??I      ??a??,?)? ?i?V?H|????Unknown
XJHostCast"Cast_5(1      ??9      ??A      ??I      ??a??,?)? ?i?i'??????Unknown
aKHostIdentity"Identity(1      ??9      ??A      ??I      ??a??,?)? ?ib|?????Unknown?
?LHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a??,?)? ?i.?]0?????Unknown
uMHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??,?)? ?i???Ҷ????Unknown
wNHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??,?)? ?iƴ?u?????Unknown
yOHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??,?)? ?i??.?????Unknown
uPHostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??a??,?)? ?i^?ɺ?????Unknown
wQHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a??,?)? ?i*?d]?????Unknown
?RHostSigmoidGrad"8gradient_tape/sequential_25/dense_80/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a??,?)? ?i?????????Unknown
?SHostReadVariableOp",sequential_25/dense_80/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??,?)? ?ia?MQ? ???Unknown*?O
uHostFlushSummaryWriter"FlushSummaryWriter(1     (?@9     (?@A     (?@I     (?@a?Wr-????i?Wr-?????Unknown?
dHostDataset"Iterator::Model(1     ?F@9     ?F@A      9@I      9@a?z?z,??i!(?PG???Unknown
iHostWriteSummary"WriteSummary(1      9@9      9@A      9@I      9@a?z?z,??i??Iڳ???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      8@9      8@A      7@I      7@aG?	?f=??i
H2?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@au?a??V??iVUUUUU???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      3@9      3@A      3@I      3@a/?)?>_??i??6JO????Unknown
lHostIteratorGetNext"IteratorGetNext(1      3@9      3@A      3@I      3@a/?)?>_??i???I{???Unknown
uHost_FusedMatMul"sequential_25/dense_78/Relu(1      2@9      2@A      2@I      2@a?~???g??i?z??????Unknown
\	HostGreater"Greater(1      *@9      *@A      *@I      *@a)??$??i3C?k???Unknown

HostMatMul"+gradient_tape/sequential_25/dense_79/MatMul(1      $@9      $@A      $@I      $@au?a??V??i?ɟr????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?~???g??iՏ??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?~???g??i?U???C???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      "@9      "@A      "@I      "@a?~???g??i???N????Unknown
?HostMatMul"-gradient_tape/sequential_25/dense_79/MatMul_1(1      "@9      "@A      "@I      "@a?~???g??i???b?????Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a?~???g??iŧ.5????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??A?~?i?<?nR???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a??A?~?ig?J9Q????Unknown
HostMatMul"+gradient_tape/sequential_25/dense_80/MatMul(1       @9       @A       @I       @a??A?~?i??X?3????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?S"?{?i_?$?Y???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?S"?{?iA??:???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_25/dense_80/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?S"?{?i???P?p???Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?S"?{?iTʉ?̦???Unknown
xHost_FusedMatMul"sequential_25/dense_80/BiasAdd(1      @9      @A      @I      @a?S"?{?i?V??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??A??4w?i????\???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??A??4w?i?kw?9???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a??A??4w?i???X0h???Unknown
HostMatMul"+gradient_tape/sequential_25/dense_78/MatMul(1      @9      @A      @I      @a??A??4w?i??:?????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @au?a??Vs?iB???G????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @au?a??Vs?i??]?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @au?a??Vs?i?hZ??
???Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @au?a??Vs?i;,?P1???Unknown
? HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @au?a??Vs?i????W???Unknown
?!HostBiasAddGrad"8gradient_tape/sequential_25/dense_78/BiasAdd/BiasAddGrad(1      @9      @A      @I      @au?a??Vs?i??4??~???Unknown
?"HostBiasAddGrad"8gradient_tape/sequential_25/dense_79/BiasAdd/BiasAddGrad(1      @9      @A      @I      @au?a??Vs?i4v}3Y????Unknown
u#HostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @au?a??Vs?i?9??????Unknown
u$Host_FusedMatMul"sequential_25/dense_79/Relu(1      @9      @A      @I      @au?a??Vs?i??V?????Unknown
u%HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a??A?n?i???????Unknown
i&HostMean"mean_squared_error/Mean(1      @9      @A      @I      @a??A?n?i,ؖ0???Unknown
?'HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a??A?n?i?$?O???Unknown
?(HostReadVariableOp"-sequential_25/dense_78/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??A?n?i~+Zyn???Unknown
?)HostReadVariableOp"-sequential_25/dense_80/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??A?n?i'
2?j????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a??A??4g?i&L???????Unknown
V+HostCast"Cast(1      @9      @A      @I      @a??A??4g?i%??|Ի???Unknown
X,HostCast"Cast_3(1      @9      @A      @I      @a??A??4g?i$Ёm	????Unknown
`-HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a??A??4g?i#G^>????Unknown
b.HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??A??4g?i"TOs???Unknown
}/HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @a??A??4g?i!???????Unknown
?0HostReluGrad"-gradient_tape/sequential_25/dense_79/ReluGrad(1      @9      @A      @I      @a??A??4g?i ؖ0?/???Unknown
?1HostMatMul"-gradient_tape/sequential_25/dense_80/MatMul_1(1      @9      @A      @I      @a??A??4g?i\!G???Unknown
?2HostReadVariableOp"-sequential_25/dense_79/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??A??4g?i\!G^???Unknown
s3HostSigmoid"sequential_25/dense_80/Sigmoid(1      @9      @A      @I      @a??A??4g?i??|u???Unknown
t4HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??A?^?iqj??????Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a??A?^?iŠ?Cm????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??A?^?i"q??????Unknown
X7HostEqual"Equal(1       @9       @A       @I       @a??A?^?im???^????Unknown
V8HostMean"Mean(1       @9       @A       @I       @a??A?^?i?$x%?????Unknown
T9HostMul"Mul(1       @9       @A       @I       @a??A?^?i???O????Unknown
s:HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a??A?^?ii'f?????Unknown
|;HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??A?^?i??A????Unknown
w<HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??A?^?i*??? ???Unknown
?=HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a??A?^?ie?	H2???Unknown
w>HostCast"%gradient_tape/mean_squared_error/Cast(1       @9       @A       @I       @a??A?^?i?,??????Unknown
u?HostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @a??A?^?i??#/???Unknown
@HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1       @9       @A       @I       @a??A?^?ia/?)?>???Unknown
uAHostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a??A?^?i???N???Unknown
}BHostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a??A?^?i	2?j?]???Unknown
?CHostReluGrad"-gradient_tape/sequential_25/dense_78/ReluGrad(1       @9       @A       @I       @a??A?^?i]?m???Unknown
?DHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??A?^?i?4??~|???Unknown
|EHostDivNoNan"&mean_squared_error/weighted_loss/value(1       @9       @A       @I       @a??A?^?i?%L?????Unknown
?FHostReadVariableOp",sequential_25/dense_78/MatMul/ReadVariableOp(1       @9       @A       @I       @a??A?^?iY7??o????Unknown
?GHostReadVariableOp",sequential_25/dense_79/MatMul/ReadVariableOp(1       @9       @A       @I       @a??A?^?i??,??????Unknown
XHHostCast"Cast_4(1      ??9      ??A      ??I      ??a??A?N?iWynݤ????Unknown
XIHostCast"Cast_5(1      ??9      ??A      ??I      ??a??A?N?i:?-a????Unknown
aJHostIdentity"Identity(1      ??9      ??A      ??I      ??a??A?N?i???}????Unknown?
?KHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a??A?N?iU?3??????Unknown
uLHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??A?N?i?{u?????Unknown
wMHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??A?N?i?<?nR????Unknown
yNHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??A?N?iS???????Unknown
uOHostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??a??A?N?i??:?????Unknown
wPHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a??A?N?i?~|_?????Unknown
?QHostSigmoidGrad"8gradient_tape/sequential_25/dense_80/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a??A?N?iQ???C????Unknown
?RHostReadVariableOp",sequential_25/dense_80/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??A?N?i?????????Unknown2CPU