"?p
BHostIDLE"IDLE1     ??@A     ??@a5/?D?)??i5/?D?)???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      m@9      m@A      c@I      c@a1???\??iE?)͋????Unknown
?HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1     ?]@9     ?]@A     @]@I     @]@a?;?????i$?Ҽ????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?X@9     ?X@A     ?X@I     ?X@aY???=??i"???????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate(1      Q@9      Q@A     ?P@I     ?P@az?'Ni??iE?)͋????Unknown
?HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1      g@9      g@A     @P@I     @P@av{?e耍?i3t?n?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@@9     ?@@A     ?@@I     ?@@ax6?;?}?i?H8?y????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1     ?@@9     ?@@A     ?@@I     ?@@ax6?;?}?i??c-???Unknown
{	HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      >@9      >@A      >@I      >@am?w6?;{?i???c???Unknown
d
HostDataset"Iterator::Model(1     ?^@9     ?^@A      8@I      8@aW?+??u?i?2t?n????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      7@9      7@A      7@I      7@aT?"?t?ibr1????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      6@9      6@A      6@I      6@aP$?Ҽ?s?iU?"????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      3@9      3@A      3@I      3@aE?)͋?q?i????????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      2@9      2@A      2@I      2@aAL? &Wp?isJ??O$???Unknown
hHostRandomShuffle"RandomShuffle(1      1@9      1@A      1@I      1@a{?e???n?i??c-C???Unknown
?HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      1@9      1@A      1@I      1@a{?e???n?i???
b???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      ,@9      ,@A      ,@I      ,@af???ki?i???v{???Unknown
XHostCast"Cast_3(1      &@9      &@A      &@I      &@aP$?Ҽ?c?i?2t?n????Unknown
^HostGatherV2"GatherV2(1      &@9      &@A      &@I      &@aP$?Ҽ?c?i??F}g????Unknown
?HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1      &@9      &@A      &@I      &@aP$?Ҽ?c?ik:`????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@aP$?Ҽ?c?iC??X????Unknown?
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@aI8?y?'b?i{?e??????Unknown
`HostGatherV2"
GatherV2_1(1      "@9      "@A      "@I      "@aAL? &W`?i?Z??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @at?n??]?i'Ni^????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @at?n??]?i????
???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @at?n??]?i???k???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1       @9       @A       @I       @at?n??]?iG8?y?'???Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1       @9       @A       @I       @at?n??]?i??l?w6???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @af???kY?i??c-C???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @af???kY?i?pJ??O???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @af???kY?i1???\???Unknown
t HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aW?+??U?i???F}g???Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aW?+??U?i??
br???Unknown
?"HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aW?+??U?i????F}???Unknown
?#HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aW?+??U?i#W?+????Unknown
?$HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aW?+??U?i? &W????Unknown
{%HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @aW?+??U?i3?;?????Unknown
r&HostTensorSliceDataset"TensorSliceDataset(1      @9      @A      @I      @aI8?y?'R?iϼ?	????Unknown
z'HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @aI8?y?'R?ik??????Unknown
v(HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aI8?y?'R?ibr1????Unknown
?)HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aI8?y?'R?i?4/?D????Unknown
?*HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aI8?y?'R?i???X????Unknown
?+HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aI8?y?'R?i?٨?l????Unknown
?,HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @at?n??M?i???????Unknown
?-HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @at?n??M?i;?pJ?????Unknown
?.HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @at?n??M?i?l?w6????Unknown
V/HostSum"Sum_2(1      @9      @A      @I      @at?n??M?i?H8?y????Unknown
?0HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @at?n??M?iK$?Ҽ????Unknown
v1HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @at?n??M?i?????????Unknown
~2HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @at?n??M?i??c-C???Unknown
v3HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @at?n??M?i[??Z????Unknown
v4HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @at?n??M?i?+?????Unknown
?5Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @at?n??M?i?n?????Unknown
?6HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @at?n??M?ikJ??O$???Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aW?+??E?i//?D?)???Unknown
r8HostConcatenateDataset"ConcatenateDataset(1      @9      @A      @I      @aW?+??E?i?	?4/???Unknown
X9HostEqual"Equal(1      @9      @A      @I      @aW?+??E?i??	?4???Unknown
e:Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aW?+??E?i{?k:???Unknown?
V;HostMean"Mean(1      @9      @A      @I      @aW?+??E?i??)͋????Unknown
s<HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aW?+??E?i?4/?D???Unknown
X=HostSlice"Slice(1      @9      @A      @I      @aW?+??E?iǋ??pJ???Unknown
Z>HostSlice"Slice_1(1      @9      @A      @I      @aW?+??E?i?pJ??O???Unknown
h?HostTensorDataset"TensorDataset(1      @9      @A      @I      @aW?+??E?iOUUUUU???Unknown
v@HostExp"%binary_crossentropy/logistic_loss/Exp(1      @9      @A      @I      @aW?+??E?i:`??Z???Unknown
`AHostDivNoNan"
div_no_nan(1      @9      @A      @I      @aW?+??E?i?k:`???Unknown
bBHostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aW?+??E?i?v{?e???Unknown
?CHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @aW?+??E?i_???k???Unknown
?DHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @aW?+??E?i#͋??p???Unknown
?EHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @aW?+??E?i籖?v???Unknown
?FHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @aW?+??E?i???v{???Unknown
?GHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aW?+??E?io{?e?????Unknown
?HHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @aW?+??E?i3`??Z????Unknown
oIHostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @aW?+??E?i?D?)͋???Unknown
vJHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @at?n??=?i?2t?n????Unknown
vKHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @at?n??=?i? &W????Unknown
VLHostCast"Cast(1       @9       @A       @I       @at?n??=?i???????Unknown
\MHostGreater"Greater(1       @9       @A       @I       @at?n??=?iW???S????Unknown
?NHostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1       @9      ??A       @I      ??at?n??=?i/?;?????Unknown
uOHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @at?n??=?i????????Unknown
|PHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @at?n??=?i?şH8????Unknown
dQHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @at?n??=?i??Q?٨???Unknown
jRHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @at?n??=?i??v{????Unknown
rSHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @at?n??=?ig??????Unknown
?THostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @at?n??=?i?}g??????Unknown
wUHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @at?n??=?ik:`????Unknown
yVHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @at?n??=?i?X??????Unknown
?WHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @at?n??=?i?F}g?????Unknown
xXHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @at?n??=?i?4/?D????Unknown
~YHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @at?n??=?iw"???????Unknown
?ZHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @at?n??=?iO?+?????Unknown
?[HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @at?n??=?i'?D?)????Unknown
?\HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @at?n??=?i???X?????Unknown
?]HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @at?n??=?i?٨?l????Unknown
~^HostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @at?n??=?i??Z?????Unknown
_HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @at?n??=?i???????Unknown
?`HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @at?n??=?i_???Q????Unknown
?aHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @at?n??=?i7?pJ?????Unknown
?bHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @at?n??=?i"??????Unknown
XcHostCast"Cast_4(1      ??9      ??A      ??I      ??at?n??-?i?u{?e????Unknown
XdHostCast"Cast_5(1      ??9      ??A      ??I      ??at?n??-?i?l?w6????Unknown
?eHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor(1      ??9      ??A      ??I      ??at?n??-?i?c-C????Unknown
?fHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??at?n??-?i?Z??????Unknown
TgHostMul"Mul(1      ??9      ??A      ??I      ??at?n??-?i?Q?٨????Unknown
}hHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??at?n??-?i?H8?y????Unknown
uiHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??at?n??-?i???pJ????Unknown
wjHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??at?n??-?io6?;????Unknown
?kHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??at?n??-?i[-C?????Unknown
?lHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??at?n??-?iG$?Ҽ????Unknown
?mHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??at?n??-?i3???????Unknown
?nHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??at?n??-?iNi^????Unknown
?oHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??at?n??-?i	?4/????Unknown
}pHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??at?n??-?i?????????Unknown
4qHostIdentity"Identity(i?????????Unknown?
irHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
[sHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
[tHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown*?o
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      m@9      m@A      c@I      c@a??????i???????Unknown
?HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1     ?]@9     ?]@A     @]@I     @]@a??i ?F?F???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?X@9     ?X@A     ?X@I     ?X@a>?>???idPdP???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate(1      Q@9      Q@A     ?P@I     ?P@aZZ??iZ?1Y?1???Unknown
?HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1      g@9      g@A     @P@I     @P@a??i???????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?@@9     ?@@A     ?@@I     ?@@a??????i?????????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1     ?@@9     ?@@A     ?@@I     ?@@a??????iQdPd???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      >@9      >@A      >@I      >@a?͛?i?}B?}B???Unknown
d	HostDataset"Iterator::Model(1     ?^@9     ?^@A      8@I      8@a?=?=??i?j??j????Unknown
}
HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      7@9      7@A      7@I      7@adPdP??i?ힲ?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      6@9      6@A      6@I      6@a(c(c??i?B?B???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      3@9      3@A      3@I      3@at?t???i??Γ?????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      2@9      2@A      2@I      2@a8?8???iWTTTTT???Unknown
hHostRandomShuffle"RandomShuffle(1      1@9      1@A      1@I      1@a ??????i8\?4\????Unknown
?HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      1@9      1@A      1@I      1@a ??????idPdP???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      ,@9      ,@A      ,@I      ,@a??????iX.?U.????Unknown
XHostCast"Cast_3(1      &@9      &@A      &@I      &@a(c(c??i??	??	???Unknown
^HostGatherV2"GatherV2(1      &@9      &@A      &@I      &@a(c(c??i?G[?G[???Unknown
?HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1      &@9      &@A      &@I      &@a(c(c??i8Ԭ6Ԭ???Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a(c(c??i?`??`????Unknown?
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a??????i??H??H???Unknown
`HostGatherV2"
GatherV2_1(1      "@9      "@A      "@I      "@a8?8???ix<?w<????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a????}?ix??w?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a????}?ix?x????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a????}?ix)=x)=???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1       @9       @A       @I       @a????}?ixxxxxx???Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1       @9       @A       @I       @a????}?ixǳxǳ???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a????y?i??瘬????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a????y?i???????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a????y?i?vO?vO???Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?=?=v?i?{?{???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?=?=v?iXm?Ym????Unknown
?!HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?=?=v?i??ԙ?????Unknown
?"HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?=?=v?i?c?c???Unknown
?#HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?=?=v?i?-?-???Unknown
{$HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a?=?=v?iXZZZZZ???Unknown
r%HostTensorSliceDataset"TensorSliceDataset(1      @9      @A      @I      @a????r?i?k?k???Unknown
z&HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a????r?i}?}????Unknown
v'HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a????r?ix??z?????Unknown
?(HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a????r?i؟?ڟ????Unknown
?)HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????r?i8?;????Unknown
?*HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????r?i??8??8???Unknown
?+HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a????m?ijVjV???Unknown
?,HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a????m?i?t?t???Unknown
?-HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a????m?i???????Unknown
V.HostSum"Sum_2(1      @9      @A      @I      @a????m?i?`??`????Unknown
?/HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a????m?i?????Unknown
v0HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a????m?i??ꛯ????Unknown
~1HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a????m?iWW???Unknown
v2HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a????m?i??%??%???Unknown
v3HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a????m?i?C?C???Unknown
?4Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a????m?i?Ma?Ma???Unknown
?5HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a????m?i?~?~???Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?=?=f?i?2??2????Unknown
r7HostConcatenateDataset"ConcatenateDataset(1      @9      @A      @I      @a?=?=f?iXp?\p????Unknown
X8HostEqual"Equal(1      @9      @A      @I      @a?=?=f?i?????????Unknown
e9Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?=?=f?i??ל?????Unknown?
V:HostMean"Mean(1      @9      @A      @I      @a?=?=f?i8)?<)????Unknown
s;HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?=?=f?i?f?f???Unknown
X<HostSlice"Slice(1      @9      @A      @I      @a?=?=f?ix?}????Unknown
Z=HostSlice"Slice_1(1      @9      @A      @I      @a?=?=f?i?0?0???Unknown
h>HostTensorDataset"TensorDataset(1      @9      @A      @I      @a?=?=f?i?G?G???Unknown
v?HostExp"%binary_crossentropy/logistic_loss/Exp(1      @9      @A      @I      @a?=?=f?iX]]]]]???Unknown
`@HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?=?=f?i??s??s???Unknown
bAHostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?=?=f?i?؉?؉???Unknown
?BHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?=?=f?i8?=????Unknown
?CHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a?=?=f?i?S??S????Unknown
?DHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?=?=f?ix??}?????Unknown
?EHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a?=?=f?i???????Unknown
?FHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?=?=f?i???????Unknown
?GHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a?=?=f?iXJ^J???Unknown
oHHostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @a?=?=f?i??%??%???Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a????]?i?[4?[4???Unknown
vJHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a????]?ix/C~/C???Unknown
VKHostCast"Cast(1       @9       @A       @I       @a????]?i8R>R???Unknown
\LHostGreater"Greater(1       @9       @A       @I       @a????]?i??`??`???Unknown
?MHostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1       @9      ??A       @I      ??a????]?i??o??o???Unknown
uNHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a????]?ix~~~~~???Unknown
|OHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a????]?i8R?>R????Unknown
dPHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a????]?i?%??%????Unknown
jQHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a????]?i?????????Unknown
rRHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a????]?ix͹~͹???Unknown
?SHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a????]?i8??>?????Unknown
wTHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a????]?i?t??t????Unknown
yUHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a????]?i?H??H????Unknown
?VHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a????]?ix?~????Unknown
xWHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a????]?i8??????Unknown
~XHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a????]?i???????Unknown
?YHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a????]?i??!??!???Unknown
?ZHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a????]?ixk0k0???Unknown
?[HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @a????]?i8????????Unknown
?\HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a????]?i?N?N???Unknown
~]HostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a????]?i??\??\???Unknown
^HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a????]?ix?k?k???Unknown
?_HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a????]?i8?z??z???Unknown
?`HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1       @9       @A       @I       @a????]?i?a??a????Unknown
?aHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a????]?i?5??5????Unknown
XbHostCast"Cast_4(1      ??9      ??A      ??I      ??a????M?i?????????Unknown
XcHostCast"Cast_5(1      ??9      ??A      ??I      ??a????M?ix	?	????Unknown
?dHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor(1      ??9      ??A      ??I      ??a????M?iXs?_s????Unknown
?eHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??a????M?i8ݵ?ݵ???Unknown
TfHostMul"Mul(1      ??9      ??A      ??I      ??a????M?iG?G????Unknown
}gHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a????M?i?????????Unknown
uhHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a????M?i???????Unknown
wiHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????M?i??ӿ?????Unknown
?jHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a????M?i??ڟ?????Unknown
?kHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a????M?ixX?X????Unknown
?lHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a????M?iX??_?????Unknown
?mHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a????M?i8,??,????Unknown
?nHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a????M?i???????Unknown
}oHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a????M?i?????????Unknown
4pHostIdentity"Identity(i?????????Unknown?
iqHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
[rHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
[sHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown2CPU