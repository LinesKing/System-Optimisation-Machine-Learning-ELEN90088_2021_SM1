"?P
BHostIDLE"IDLE1     ??@A     ??@a??Ώ??i??Ώ???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a?)-z?i??i?]^}???Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      :@9      :@A      :@I      :@a??s?h?i??\ҿ????Unknown
iHostWriteSummary"WriteSummary(1      :@9      :@A      :@I      :@a??s?h?iL?[Fx????Unknown?
uHost_FusedMatMul"sequential_18/dense_57/Relu(1      7@9      7@A      7@I      7@a`<:??e?i????V????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@a?&lu
c?i?h?Z????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A      1@I      1@a?훰?)`?i??e?????Unknown
VHostCast"Cast(1      &@9      &@A      &@I      &@a???M??T?i= ???????Unknown
?	HostMatMul"-gradient_tape/sequential_18/dense_58/MatMul_1(1      $@9      $@A      $@I      $@a?&lu
S?iP??{????Unknown
l
HostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@aIV??<Q?i?&lu
???Unknown
eHost
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@aIV??<Q?i???????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aIV??<Q?iQ	?'???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aIV??<Q?i?xWP????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      "@9      "@A      "@I      "@aIV??<Q?i????D&???Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a,???lN?i?&?-???Unknown
dHostDataset"Iterator::Model(1      <@9      <@A       @I       @a,???lN?i-@j]{5???Unknown
HostMatMul"+gradient_tape/sequential_18/dense_59/MatMul(1       @9       @A       @I       @a,???lN?ipk̔=???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a,???lN?i??.̱D???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?i??A?J?i?|??YK???Unknown
HostMatMul"+gradient_tape/sequential_18/dense_58/MatMul(1      @9      @A      @I      @a?i??A?J?igbmR???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @aaȁ&??F?i??ֵW???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aaȁ&??F?iK?-@j]???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_18/dense_59/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aaȁ&??F?i?C??c???Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aaȁ&??F?i/?@?h???Unknown
XHostCast"Cast_5(1      @9      @A      @I      @a?&lu
C?i9???m???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?&lu
C?iC?{Ur???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?&lu
C?iM?w???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?&lu
C?iWP??{???Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a?&lu
C?ia?S ?????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a,???l>?i??e????Unknown
\HostGreater"Greater(1      @9      @A      @I      @a,???l>?i?ֵW3????Unknown
? HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a,???l>?iD?f? ????Unknown
?!HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a,???l>?i??Ώ???Unknown
?"HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a,???l>?i??*?????Unknown
?#HostBiasAddGrad"8gradient_tape/sequential_18/dense_57/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a,???l>?i'-z?i????Unknown
$HostMatMul"+gradient_tape/sequential_18/dense_57/MatMul(1      @9      @A      @I      @a,???l>?i?B+b7????Unknown
?%HostMatMul"-gradient_tape/sequential_18/dense_59/MatMul_1(1      @9      @A      @I      @a,???l>?iiX??????Unknown
?&HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a,???l>?i
n??Ң???Unknown
u'Host_FusedMatMul"sequential_18/dense_58/Relu(1      @9      @A      @I      @a,???l>?i??>5?????Unknown
x(Host_FusedMatMul"sequential_18/dense_59/BiasAdd(1      @9      @A      @I      @a,???l>?iL???m????Unknown
?)HostReadVariableOp"-sequential_18/dense_59/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a,???l>?i???l;????Unknown
t*HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aaȁ&??6?i&e?????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aaȁ&??6?i_O*??????Unknown
X,HostCast"Cast_4(1      @9      @A      @I      @aaȁ&??6?i??
ʶ???Unknown
T-HostMul"Mul(1      @9      @A      @I      @aaȁ&??6?i?????????Unknown
s.HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aaȁ&??6?i
?xt~????Unknown
u/HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aaȁ&??6?iC?=?X????Unknown
}0HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @aaȁ&??6?i|`?2????Unknown
?1HostReluGrad"-gradient_tape/sequential_18/dense_57/ReluGrad(1      @9      @A      @I      @aaȁ&??6?i?0?????Unknown
?2HostBiasAddGrad"8gradient_tape/sequential_18/dense_58/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aaȁ&??6?i? ?G?????Unknown
i3HostMean"mean_squared_error/Mean(1      @9      @A      @I      @aaȁ&??6?i'?P|?????Unknown
?4HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      @9      @A      @I      @aaȁ&??6?i`???????Unknown
?5HostReadVariableOp",sequential_18/dense_57/MatMul/ReadVariableOp(1      @9      @A      @I      @aaȁ&??6?i?q??u????Unknown
?6HostReadVariableOp"-sequential_18/dense_58/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aaȁ&??6?i?A?P????Unknown
?7HostReadVariableOp",sequential_18/dense_58/MatMul/ReadVariableOp(1      @9      @A      @I      @aaȁ&??6?idO*????Unknown
s8HostSigmoid"sequential_18/dense_59/Sigmoid(1      @9      @A      @I      @aaȁ&??6?iD?(?????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a,???l.?imR?????Unknown
X:HostCast"Cast_3(1       @9       @A       @I       @a,???l.?i????????Unknown
X;HostEqual"Equal(1       @9       @A       @I       @a,???l.?i?????????Unknown
V<HostMean"Mean(1       @9       @A       @I       @a,???l.?i????????Unknown
`=HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a,???l.?iY?c??????Unknown
u>HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a,???l.?i*#<Wm????Unknown
b?HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a,???l.?i??%T????Unknown
w@HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a,???l.?i?8??:????Unknown
wAHostCast"%gradient_tape/mean_squared_error/Cast(1       @9       @A       @I       @a,???l.?i????!????Unknown
uBHostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @a,???l.?inN??????Unknown
uCHostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @a,???l.?i??v\?????Unknown
uDHostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a,???l.?idO*?????Unknown
}EHostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a,???l.?i??'??????Unknown
?FHostReluGrad"-gradient_tape/sequential_18/dense_58/ReluGrad(1       @9       @A       @I       @a,???l.?i?y ƣ????Unknown
|GHostDivNoNan"&mean_squared_error/weighted_loss/value(1       @9       @A       @I       @a,???l.?i?ٓ?????Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a,???l?i?I??}????Unknown
aIHostIdentity"Identity(1      ??9      ??A      ??I      ??a,???l?iS??aq????Unknown?
?JHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a,???l?i?ԝ?d????Unknown
|KHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a,???l?i#?/X????Unknown
wLHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a,???l?i?_v?K????Unknown
yMHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a,???l?i??b?>????Unknown
?NHostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??a,???l?i[?Nd2????Unknown
OHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      ??9      ??A      ??I      ??a,???l?i?/;?%????Unknown
wPHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a,???l?i+u'2????Unknown
?QHostSigmoidGrad"8gradient_tape/sequential_18/dense_59/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a,???l?i???????Unknown
?RHostReadVariableOp"-sequential_18/dense_57/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a,???l?i?????????Unknown
?SHostReadVariableOp",sequential_18/dense_59/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a,???l?i?"v?y ???Unknown*?O
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a*?A?g??i*?A?g???Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      :@9      :@A      :@I      :@ad!Y?B??i5F
?}????Unknown
iHostWriteSummary"WriteSummary(1      :@9      :@A      :@I      :@ad!Y?B??i@???y???Unknown?
uHost_FusedMatMul"sequential_18/dense_57/Relu(1      7@9      7@A      7@I      7@a;?;???i)p????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@a?TWέ??i???e????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A      1@I      1@a???A??i3?*j????Unknown
VHostCast"Cast(1      &@9      &@A      &@I      &@a?C??ւ?iB??_???Unknown
?HostMatMul"-gradient_tape/sequential_18/dense_58/MatMul_1(1      $@9      $@A      $@I      $@a?TWέ??i?yVQc????Unknown
l	HostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a??6@??~?i-?֏????Unknown
e
Host
LogicalAnd"
LogicalAnd(1      "@9      "@A      "@I      "@a??6@??~?i?TWέ???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a??6@??~?i]??S]???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a??6@??~?i?/XK?????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      "@9      "@A      "@I      "@a??6@??~?i??؉?????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a????e{?ii?Oi???Unknown
dHostDataset"Iterator::Model(1      <@9      <@A       @I       @a????e{?iE?g5F???Unknown
HostMatMul"+gradient_tape/sequential_18/dense_59/MatMul(1       @9       @A       @I       @a????e{?i!/? }???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a????e{?i????̳???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aDG?&?w?i#??????Unknown
HostMatMul"+gradient_tape/sequential_18/dense_58/MatMul(1      @9      @A      @I      @aDG?&?w?i??;????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?2?*j?t?i?Oi?<???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?2?*j?t?i	????e???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_18/dense_59/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?2?*j?t?in???????Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?2?*j?t?i?*j?????Unknown
XHostCast"Cast_5(1      @9      @A      @I      @a?TWέq?i}??S????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?TWέq?i'??C?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?TWέq?i?6@?????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?TWέq?i{???A???Unknown
uHostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a?TWέq?i%?yVQc???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a????ek?iS]9?~???Unknown
\HostGreater"Greater(1      @9      @A      @I      @a????ek?iA????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a????ek?i??$??????Unknown
? HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a????ek?iݏ??????Unknown
?!HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a????ek?i?N??N????Unknown
?"HostBiasAddGrad"8gradient_tape/sequential_18/dense_57/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????ek?i?Ч????Unknown
#HostMatMul"+gradient_tape/sequential_18/dense_57/MatMul(1      @9      @A      @I      @a????ek?i?̳?#???Unknown
?$HostMatMul"-gradient_tape/sequential_18/dense_59/MatMul_1(1      @9      @A      @I      @a????ek?i???m?>???Unknown
?%HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a????ek?i?J{P?Y???Unknown
u&Host_FusedMatMul"sequential_18/dense_58/Relu(1      @9      @A      @I      @a????ek?iq	_3Lu???Unknown
x'Host_FusedMatMul"sequential_18/dense_59/BiasAdd(1      @9      @A      @I      @a????ek?i_?B?????Unknown
?(HostReadVariableOp"-sequential_18/dense_59/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????ek?iM?&?????Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?2?*j?d?i?VQc?????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?2?*j?d?i?%|?0????Unknown
X+HostCast"Cast_4(1      @9      @A      @I      @a?2?*j?d?i???7?????Unknown
T,HostMul"Mul(1      @9      @A      @I      @a?2?*j?d?i?ѡI????Unknown
s-HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?2?*j?d?iL??????Unknown
u.HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?2?*j?d?ib'vb'???Unknown
}/HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @a?2?*j?d?i?1R??;???Unknown
?0HostReluGrad"-gradient_tape/sequential_18/dense_57/ReluGrad(1      @9      @A      @I      @a?2?*j?d?i? }J{P???Unknown
?1HostBiasAddGrad"8gradient_tape/sequential_18/dense_58/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?2?*j?d?iЧ?e???Unknown
i2HostMean"mean_squared_error/Mean(1      @9      @A      @I      @a?2?*j?d?iK???y???Unknown
?3HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a?2?*j?d?i~n?? ????Unknown
?4HostReadVariableOp",sequential_18/dense_57/MatMul/ReadVariableOp(1      @9      @A      @I      @a?2?*j?d?i?=(??????Unknown
?5HostReadVariableOp"-sequential_18/dense_58/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?2?*j?d?i?S]9????Unknown
?6HostReadVariableOp",sequential_18/dense_58/MatMul/ReadVariableOp(1      @9      @A      @I      @a?2?*j?d?i?}??????Unknown
s7HostSigmoid"sequential_18/dense_59/Sigmoid(1      @9      @A      @I      @a?2?*j?d?iJ??1R????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a????e[?i??#????Unknown
X9HostCast"Cast_3(1       @9       @A       @I       @a????e[?i8j??????Unknown
X:HostEqual"Equal(1       @9       @A       @I       @a????e[?i?I?k	???Unknown
V;HostMean"Mean(1       @9       @A       @I       @a????e[?i&)p????Unknown
`<HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a????e[?i????$???Unknown
u=HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a????e[?i?Sڃ2???Unknown
b>HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a????e[?i????6@???Unknown
w?HostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a????e[?i?7??M???Unknown
w@HostCast"%gradient_tape/mean_squared_error/Cast(1       @9       @A       @I       @a????e[?iy????[???Unknown
uAHostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @a????e[?i?e?Oi???Unknown
uBHostSum"$gradient_tape/mean_squared_error/Sum(1       @9       @A       @I       @a????e[?igE??w???Unknown
uCHostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a????e[?i?$???????Unknown
}DHostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a????e[?iUqth????Unknown
?EHostReluGrad"-gradient_tape/sequential_18/dense_58/ReluGrad(1       @9       @A       @I       @a????e[?i???e????Unknown
|FHostDivNoNan"&mean_squared_error/weighted_loss/value(1       @9       @A       @I       @a????e[?iC?TWέ???Unknown
vGHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a????eK?i??Ч????Unknown
aHHostIdentity"Identity(1      ??9      ??A      ??I      ??a????eK?i???H?????Unknown?
?IHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a????eK?iw??Z????Unknown
|JHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a????eK?i3?8:4????Unknown
wKHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????eK?i?q??????Unknown
yLHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????eK?i?a?+?????Unknown
?MHostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??a????eK?igQc??????Unknown
NHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      ??9      ??A      ??I      ??a????eK?i#A?????Unknown
wOHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a????eK?i?0Օs????Unknown
?PHostSigmoidGrad"8gradient_tape/sequential_18/dense_59/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a????eK?i? ?M????Unknown
?QHostReadVariableOp"-sequential_18/dense_57/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a????eK?iWG?&????Unknown
?RHostReadVariableOp",sequential_18/dense_59/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a????eK?i	     ???Unknown2CPU