"?O
BHostIDLE"IDLE1     ??@A     ??@aAva1J??iAva1J???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a]?k????i?&??p????Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      F@9      F@A      F@I      F@a+?|/?u?im ?<???Unknown
iHostWriteSummary"WriteSummary(1      B@9      B@A      B@I      B@a:M U?q?ia???/???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      ;@9      ;@A      ;@I      ;@a?s0??j?i{?ifaJ???Unknown
xHost_FusedMatMul"sequential_15/dense_49/BiasAdd(1      ;@9      ;@A      ;@I      ;@a?s0??j?i????e???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      6@I      6@a+?|/?e?i?>??z???Unknown
uHost_FusedMatMul"sequential_15/dense_47/Relu(1      5@9      5@A      5@I      5@an?%???d?ind?ͯ????Unknown
?	HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      4@9      4@A      4@I      4@a???B?c?i?2?}????Unknown

HostMatMul"+gradient_tape/sequential_15/dense_49/MatMul(1      3@9      3@A      3@I      3@a?mw???b?ik??L????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      2@9      2@A      2@I      2@a:M U?a?i??0????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      ,@9      ,@A      ,@I      ,@a???	??[?i????????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      ,@9      ,@A      ,@I      ,@a???	??[?iLR&?????Unknown
dHostDataset"Iterator::Model(1     ?C@9     ?C@A      (@I      (@a?+?W?i?g"??????Unknown
^HostGatherV2"GatherV2(1      $@9      $@A      $@I      $@a???B?S?i?%U?????Unknown
uHostReadVariableOp"div_no_nan/ReadVariableOp(1      $@9      $@A      $@I      $@a???B?S?ic6)?????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      $@9      $@A      $@I      $@a???B?S?i??,?m???Unknown
?HostMatMul"-gradient_tape/sequential_15/dense_48/MatMul_1(1      $@9      $@A      $@I      $@a???B?S?i?08T???Unknown
|HostDivNoNan"&mean_squared_error/weighted_loss/value(1      $@9      $@A      $@I      $@a???B?S?i8l3?:!???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a:M U?Q?i_|?$*???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a:M U?Q?i??9.3???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a??
ЮO?i?E<??:???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??
ЮO?i??>??B???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a??
ЮO?i??AJ?J???Unknown
uHost_FusedMatMul"sequential_15/dense_48/Relu(1       @9       @A       @I       @a??
ЮO?i?pD??R???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a???	??K?i???;?Y???Unknown
HostMatMul"+gradient_tape/sequential_15/dense_47/MatMul(1      @9      @A      @I      @a???	??K?ih4Iy?`???Unknown
HostMatMul"+gradient_tape/sequential_15/dense_48/MatMul(1      @9      @A      @I      @a???	??K?iM?˶?g???Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a???	??K?i2?M?tn???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?+?G?i?P?et???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?+?G?i?R?Vz???Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a???B?C?i^???I???Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a???B?C?iuU#=????Unknown
?"HostBiasAddGrad"8gradient_tape/sequential_15/dense_47/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???B?C?i?(?s0????Unknown
?#HostBiasAddGrad"8gradient_tape/sequential_15/dense_48/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???B?C?iJ?X?#????Unknown
?$HostBiasAddGrad"8gradient_tape/sequential_15/dense_49/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???B?C?i???????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a??
Ю??iq???????Unknown
?&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a??
Ю??i?H??????Unknown
u'HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a??
Ю??iw?ޢ?????Unknown
?(HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a??
Ю??i??|?????Unknown
?)HostMatMul"-gradient_tape/sequential_15/dense_49/MatMul_1(1      @9      @A      @I      @a??
Ю??i}^?V?????Unknown
i*HostMean"mean_squared_error/Mean(1      @9      @A      @I      @a??
Ю??i ??0ڪ???Unknown
?+HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a??
Ю??i??
Ю???Unknown
u,HostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a??
Ю??it??Ų???Unknown
?-HostReadVariableOp",sequential_15/dense_48/MatMul/ReadVariableOp(1      @9      @A      @I      @a??
Ю??i??澻????Unknown
?.HostReadVariableOp",sequential_15/dense_49/MatMul/ReadVariableOp(1      @9      @A      @I      @a??
Ю??i-蘱????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?+?7?in2i??????Unknown
V0HostCast"Cast(1      @9      @A      @I      @a?+?7?i?7?_?????Unknown
X1HostEqual"Equal(1      @9      @A      @I      @a?+?7?i2=kÚ????Unknown
\2HostGreater"Greater(1      @9      @A      @I      @a?+?7?i?B?&?????Unknown
s3HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?+?7?i?Gm??????Unknown
`4HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?+?7?iXM???????Unknown
b5HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?+?7?i?RoQ|????Unknown
?6HostReluGrad"-gradient_tape/sequential_15/dense_48/ReluGrad(1      @9      @A      @I      @a?+?7?iX??t????Unknown
?7HostReadVariableOp"-sequential_15/dense_49/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?+?7?i~]qm????Unknown
t8HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??
Ю/?i?rh????Unknown
X9HostCast"Cast_3(1       @9       @A       @I       @a??
Ю/?i ?r?b????Unknown
V:HostMean"Mean(1       @9       @A       @I       @a??
Ю/?iAhs?]????Unknown
T;HostMul"Mul(1       @9       @A       @I       @a??
Ю/?i?t?X????Unknown
|<HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??
Ю/?i??t?S????Unknown
w=HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??
Ю/?isu?N????Unknown
w>HostCast"%gradient_tape/mean_squared_error/Cast(1       @9       @A       @I       @a??
Ю/?iE!v?I????Unknown
}?HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a??
Ю/?i??v?D????Unknown
u@HostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @a??
Ю/?i?}wm?????Unknown
uAHostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @a??
Ю/?i,xZ:????Unknown
uBHostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a??
Ю/?iI?xG5????Unknown
}CHostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a??
Ю/?i??y40????Unknown
?DHostSigmoidGrad"8gradient_tape/sequential_15/dense_49/Sigmoid/SigmoidGrad(1       @9       @A       @I       @a??
Ю/?i?6z!+????Unknown
?EHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??
Ю/?i?z&????Unknown
?FHostReadVariableOp"-sequential_15/dense_47/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??
Ю/?iM?{? ????Unknown
?GHostReadVariableOp"-sequential_15/dense_48/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??
Ю/?i?A|?????Unknown
sHHostSigmoid"sequential_15/dense_49/Sigmoid(1       @9       @A       @I       @a??
Ю/?i??|?????Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??
Ю?i?F?K????Unknown
XJHostCast"Cast_4(1      ??9      ??A      ??I      ??a??
Ю?i?}?????Unknown
XKHostCast"Cast_5(1      ??9      ??A      ??I      ??a??
Ю?i2??8????Unknown
aLHostIdentity"Identity(1      ??9      ??A      ??I      ??a??
Ю?iSL~?????Unknown?
yMHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??
Ю?it??%
????Unknown
?NHostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??a??
Ю?i??~?????Unknown
OHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      ??9      ??A      ??I      ??a??
Ю?i?Q?????Unknown
wPHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a??
Ю?iר?????Unknown
?QHostReluGrad"-gradient_tape/sequential_15/dense_47/ReluGrad(1      ??9      ??A      ??I      ??a??
Ю?i?????????Unknown
?RHostReadVariableOp",sequential_15/dense_47/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??
Ю?i?+@?~ ???Unknown
JSHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?+@?~ ???Unknown*?O
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a???	ۏ??i???	ۏ???Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      F@9      F@A      F@I      F@aJ?4?{??i?R?????Unknown
iHostWriteSummary"WriteSummary(1      B@9      B@A      B@I      B@a?X??=??i÷?z????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      ;@9      ;@A      ;@I      ;@a???N???i??x?^???Unknown
xHost_FusedMatMul"sequential_15/dense_49/BiasAdd(1      ;@9      ;@A      ;@I      ;@a???N???i?x?^???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      6@I      6@aJ?4?{??i??1q7????Unknown
uHost_FusedMatMul"sequential_15/dense_47/Relu(1      5@9      5@A      5@I      5@a??=3???i>?˒W5???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      4@9      4@A      4@I      4@aZ1G??̐?i??EQ?????Unknown
	HostMatMul"+gradient_tape/sequential_15/dense_49/MatMul(1      3@9      3@A      3@I      3@a?ݠ?????i@r??n;???Unknown
?
HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      2@9      2@A      2@I      2@a?X??=??i??ۤe????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      ,@9      ,@A      ,@I      ,@aE??Z???i?4?{???Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      ,@9      ,@A      ,@I      ,@aE??Z???i?)S|?p???Unknown
dHostDataset"Iterator::Model(1     ?C@9     ?C@A      (@I      (@a9;"_))??i???!5????Unknown
^HostGatherV2"GatherV2(1      $@9      $@A      $@I      $@aZ1G??̀?i}?i???Unknown
uHostReadVariableOp"div_no_nan/ReadVariableOp(1      $@9      $@A      $@I      $@aZ1G??̀?iB?I??G???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      $@9      $@A      $@I      $@aZ1G??̀?i	??Њ???Unknown
?HostMatMul"-gradient_tape/sequential_15/dense_48/MatMul_1(1      $@9      $@A      $@I      $@aZ1G??̀?i?%Ğ????Unknown
|HostDivNoNan"&mean_squared_error/weighted_loss/value(1      $@9      $@A      $@I      $@aZ1G??̀?i?B~8???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a?X??=~?iC???M???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a?X??=~?i?<v/????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?N?~??z?i??9??????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?N?~??z?i1q7??????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1       @9       @A       @I       @a?N?~??z?i?!5?x+???Unknown
uHost_FusedMatMul"sequential_15/dense_48/Relu(1       @9       @A       @I       @a?N?~??z?im?2?;a???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aE??Z?w?i???F????Unknown
HostMatMul"+gradient_tape/sequential_15/dense_47/MatMul(1      @9      @A      @I      @aE??Z?w?i???EQ????Unknown
HostMatMul"+gradient_tape/sequential_15/dense_48/MatMul(1      @9      @A      @I      @aE??Z?w?i???[????Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aE??Z?w?i????f???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a9;"_))t?ii?E???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a9;"_))t?i?E'Wn???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aZ1G???p?i???F?????Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aZ1G???p?iGbd6?????Unknown
?!HostBiasAddGrad"8gradient_tape/sequential_15/dense_47/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aZ1G???p?i??&?????Unknown
?"HostBiasAddGrad"8gradient_tape/sequential_15/dense_48/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aZ1G???p?i?s????Unknown
?#HostBiasAddGrad"8gradient_tape/sequential_15/dense_49/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aZ1G???p?ip@???Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a?N?~??j?i?徑?0???Unknown
?%HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?N?~??j?i?=?K???Unknown
u&HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?N?~??j?i]????f???Unknown
?'HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?N?~??j?i?n;7?????Unknown
?(HostMatMul"-gradient_tape/sequential_15/dense_49/MatMul_1(1      @9      @A      @I      @a?N?~??j?i?F??t????Unknown
i)HostMean"mean_squared_error/Mean(1      @9      @A      @I      @a?N?~??j?iJ9PV????Unknown
?*HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a?N?~??j?i????7????Unknown
u+HostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a?N?~??j?i??6i????Unknown
?,HostReadVariableOp",sequential_15/dense_48/MatMul/ReadVariableOp(1      @9      @A      @I      @a?N?~??j?i7???????Unknown
?-HostReadVariableOp",sequential_15/dense_49/MatMul/ReadVariableOp(1      @9      @A      @I      @a?N?~??j?i??4??"???Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a9;"_))d?i????7???Unknown
V/HostCast"Cast(1      @9      @A      @I      @a9;"_))d?i????.K???Unknown
X0HostEqual"Equal(1      @9      @A      @I      @a9;"_))d?i7?Q?W_???Unknown
\1HostGreater"Greater(1      @9      @A      @I      @a9;"_))d?ir	?'?s???Unknown
s2HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a9;"_))d?i?+Q?????Unknown
`3HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a9;"_))d?i?Mozӛ???Unknown
b4HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a9;"_))d?i#pΣ?????Unknown
?5HostReluGrad"-gradient_tape/sequential_15/dense_48/ReluGrad(1      @9      @A      @I      @a9;"_))d?i^?-?%????Unknown
?6HostReadVariableOp"-sequential_15/dense_49/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a9;"_))d?i????N????Unknown
t7HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?N?~??Z?i? ̼?????Unknown
X8HostCast"Cast_3(1       @9       @A       @I       @a?N?~??Z?i???0????Unknown
V9HostMean"Mean(1       @9       @A       @I       @a?N?~??Z?i?JI? ???Unknown
T:HostMul"Mul(1       @9       @A       @I       @a?N?~??Z?i5e????Unknown
|;HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?N?~??Z?i\??Ղ???Unknown
w<HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?N?~??Z?i?=	??(???Unknown
w=HostCast"%gradient_tape/mean_squared_error/Cast(1       @9       @A       @I       @a?N?~??Z?i??Hbd6???Unknown
}>HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a?N?~??Z?i??(?C???Unknown
u?HostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @a?N?~??Z?i????EQ???Unknown
u@HostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @a?N?~??Z?i???^???Unknown
uAHostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a?N?~??Z?iFZF{'l???Unknown
}BHostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a?N?~??Z?imƅA?y???Unknown
?CHostSigmoidGrad"8gradient_tape/sequential_15/dense_49/Sigmoid/SigmoidGrad(1       @9       @A       @I       @a?N?~??Z?i?2?	????Unknown
?DHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?N?~??Z?i???y????Unknown
?EHostReadVariableOp"-sequential_15/dense_47/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?N?~??Z?i?
D??????Unknown
?FHostReadVariableOp"-sequential_15/dense_48/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?N?~??Z?i	w?Z[????Unknown
sGHostSigmoid"sequential_15/dense_49/Sigmoid(1       @9       @A       @I       @a?N?~??Z?i0?? ̼???Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?N?~??J?iD?⃄????Unknown
XIHostCast"Cast_4(1      ??9      ??A      ??I      ??a?N?~??J?iXO?<????Unknown
XJHostCast"Cast_5(1      ??9      ??A      ??I      ??a?N?~??J?il"J?????Unknown
aKHostIdentity"Identity(1      ??9      ??A      ??I      ??a?N?~??J?i??A??????Unknown?
yLHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?N?~??J?i?qaf????Unknown
?MHostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??a?N?~??J?i?'?s????Unknown
NHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      ??9      ??A      ??I      ??a?N?~??J?i?ݠ??????Unknown
wOHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a?N?~??J?iГ?9?????Unknown
?PHostReluGrad"-gradient_tape/sequential_15/dense_47/ReluGrad(1      ??9      ??A      ??I      ??a?N?~??J?i?I??G????Unknown
?QHostReadVariableOp",sequential_15/dense_47/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?N?~??J?i?????????Unknown
JRHostReadVariableOp"div_no_nan/ReadVariableOp_1(i?????????Unknown2CPU