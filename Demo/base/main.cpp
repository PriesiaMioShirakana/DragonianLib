#include <iostream>

#include "Libraries/AvCodec/AvCodec.h"
#include "Libraries/Stft/Stft.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

#ifndef DRAGONIANLIB_USE_SHARED_LIBS

#include "TensorLib/Include/Base/Tensor/Tensor.h"

#endif

int main()
{
#ifndef DRAGONIANLIB_USE_SHARED_LIBS
	const int size = 10;
	auto [audio, sr] = DragonianLib::AvCodec::OpenInputStream(LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals.wav)").DecodeAudio(1);

	auto v = audio.View(1, 1, -1);
	v = v.Interpolate<DragonianLib::Operators::InterpolateMode::Linear>(
		DragonianLib::IDim(-1),
		DragonianLib::IScale( double(16000) / double(sr) )
	);
	v.Evaluate();
	DragonianLib::NumpyFileFormat::SaveNumpyFile(
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals.npy)",
		v.Shape(),
		v.Data(),
		v.ElementCount()
	);


	return 0;
#endif
}